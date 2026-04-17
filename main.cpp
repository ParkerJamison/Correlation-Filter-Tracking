#include <opencv2/opencv.hpp>

#include "CFT_Track.hpp"
#include "TrackID.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void fft2d(Mat& frame);
Mat guass(Mat initFrame, Rect2d rect, int sigma);
void trainFilter(Mat initFrame, Mat& G, vector<Mat>& A, vector<Mat>& B,
                 int numTrain, double lr);

namespace {
constexpr double kLearningRate = 0.125;
constexpr int kSigma = 100;
constexpr int kNumTrain = 16;
constexpr int kMinimumTrackSize = 8;

const string kLightGlueScript = R"PY(
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(repo_root / "LightGlue"))

from lightglue import LightGlue, SIFT
from lightglue.utils import load_image, rbd

reference_path = Path(sys.argv[1])
frame_path = Path(sys.argv[2])

device = "cpu"
extractor = SIFT(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="sift").eval().to(device)

image0 = load_image(reference_path).to(device)
image1 = load_image(frame_path).to(device)

with torch.inference_mode():
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})

feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
matches = matches01["matches"]

if matches.numel() == 0 or matches.shape[0] < 4:
    print("NO_MATCHES")
    sys.exit(0)

matched_points = feats1["keypoints"][matches[:, 1]].detach().cpu().numpy()

if matched_points.shape[0] >= 10:
    lo = np.percentile(matched_points, 10, axis=0)
    hi = np.percentile(matched_points, 90, axis=0)
else:
    lo = matched_points.min(axis=0)
    hi = matched_points.max(axis=0)

width = max(float(hi[0] - lo[0]), 1.0)
height = max(float(hi[1] - lo[1]), 1.0)
pad = max(8.0, 0.15 * max(width, height))

x = lo[0] - pad
y = lo[1] - pad
w = width + (2.0 * pad)
h = height + (2.0 * pad)

print("BBOX", int(round(x)), int(round(y)), int(round(w)), int(round(h)), int(matches.shape[0]))
)PY";

struct LightGlueMatch {
    Rect bbox;
    int matches = 0;
    string message;
};

bool isImageFile(const fs::path& path) {
    string ext = path.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp" || ext == ".webp";
}

fs::path findRepoRoot() {
    vector<fs::path> starts;

    error_code ec;
    starts.push_back(fs::current_path(ec));
    starts.push_back(fs::absolute(fs::path(__FILE__), ec).parent_path());

    for (const fs::path& start : starts) {
        if (start.empty()) continue;

        fs::path current = start;
        while (!current.empty()) {
            if (fs::exists(current / "LightGlue") && fs::exists(current / "Tracks")) {
                return current;
            }

            fs::path parent = current.parent_path();
            if (parent == current) break;
            current = parent;
        }
    }

    return fs::current_path(ec);
}

vector<fs::path> collectTrackImages(const fs::path& tracksRoot) {
    vector<fs::path> images;

    if (!fs::exists(tracksRoot)) {
        cout << "Tracks folder not found at " << tracksRoot << endl;
        return images;
    }

    for (const fs::directory_entry& entry :
         fs::recursive_directory_iterator(tracksRoot)) {
        if (entry.is_regular_file() && isImageFile(entry.path())) {
            images.push_back(entry.path());
        }
    }

    sort(images.begin(), images.end());
    return images;
}

string shellQuote(const string& value) {
    string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

string runCommand(const string& command) {
    array<char, 512> buffer{};
    string output;

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return "ERROR: failed to start command";
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    int status = pclose(pipe);
    if (status != 0 && output.empty()) {
        output = "ERROR: command exited with status " + to_string(status);
    }

    return output;
}

string summarizeOutput(const string& output) {
    constexpr size_t kMaxOutput = 1200;
    if (output.size() <= kMaxOutput) {
        return output;
    }

    return output.substr(0, kMaxOutput) + "\n... output truncated ...";
}

optional<LightGlueMatch> parseLightGlueOutput(const string& output) {
    istringstream lines(output);
    string line;

    while (getline(lines, line)) {
        if (line.rfind("BBOX ", 0) == 0) {
            istringstream values(line);
            string tag;
            int x = 0;
            int y = 0;
            int w = 0;
            int h = 0;
            int matches = 0;
            values >> tag >> x >> y >> w >> h >> matches;
            if (values && w > 0 && h > 0) {
                return LightGlueMatch{Rect(x, y, w, h), matches, line};
            }
        }

        if (line == "NO_MATCHES") {
            return nullopt;
        }
    }

    return LightGlueMatch{Rect(), 0, output};
}

optional<LightGlueMatch> findInitialBoxWithLightGlue(
    const fs::path& referenceImage,
    const fs::path& frameImage,
    const fs::path& repoRoot,
    const Size& frameSize) {

    string command = "python3 -c " + shellQuote(kLightGlueScript) + " " +
                     shellQuote(referenceImage.string()) + " " +
                     shellQuote(frameImage.string()) + " " +
                     shellQuote(repoRoot.string()) + " 2>&1";

    optional<LightGlueMatch> result = parseLightGlueOutput(runCommand(command));
    if (!result) {
        return nullopt;
    }

    if (result->bbox.empty()) {
        cout << "LightGlue command failed for " << referenceImage << ":\n"
             << summarizeOutput(result->message) << endl;
        return nullopt;
    }

    Rect frameBounds(0, 0, frameSize.width, frameSize.height);
    result->bbox &= frameBounds;

    if (result->bbox.width < kMinimumTrackSize ||
        result->bbox.height < kMinimumTrackSize) {
        return nullopt;
    }

    return result;
}

Track initTrackFromBBox(const Mat& frame, const Rect& bbox) {
    Track track;
    track.initBBox(frame, bbox);

    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
    grayFrame.convertTo(grayFrame, CV_64F);

    Mat response = guass(grayFrame, track.getSearchArea(), kSigma);
    track.G = track.cropForSearch(response);
    track.fi = track.cropForSearch(grayFrame);

    fft2d(track.G);
    trainFilter(track.fi, track.G, track.A, track.B, kNumTrain, kLearningRate);

    return track;
}

optional<Track> manualTrackInit(const Mat& frame) {
    Rect roi = selectROI("Video Player", frame, false, false);

    Rect bounds(0, 0, frame.cols, frame.rows);
    roi &= bounds;

    if (roi.width < kMinimumTrackSize || roi.height < kMinimumTrackSize) {
        return nullopt;
    }

    return initTrackFromBBox(frame, roi);
}
}  // namespace

int main() {
    fs::path repoRoot = findRepoRoot();
    fs::path tracksRoot = repoRoot / "Tracks";
    vector<fs::path> referenceImages = collectTrackImages(tracksRoot);
    vector<Track> tracks;

    CFT tracker;

    Mat currentFrame;
    namedWindow("Video Player");
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "No video stream detected" << endl;
        return 0;
    }

    waitKey(100);
    cap >> currentFrame;

    if (currentFrame.empty()) {
        cout << "Camera returned an empty initial frame" << endl;
        return 0;
    }

    fs::path tempFramePath =
        fs::temp_directory_path() /
        ("cft_lightglue_frame_" +
         to_string(chrono::steady_clock::now().time_since_epoch().count()) +
         ".jpg");

    if (!imwrite(tempFramePath.string(), currentFrame)) {
        cout << "Could not write temporary frame for LightGlue: "
             << tempFramePath << endl;
    } else {
        for (const fs::path& referenceImage : referenceImages) {
            optional<LightGlueMatch> match = findInitialBoxWithLightGlue(
                referenceImage, tempFramePath, repoRoot, currentFrame.size());

            if (!match) {
                cout << "LightGlue did not find enough matches for "
                     << referenceImage << endl;
                continue;
            }

            tracks.push_back(initTrackFromBBox(currentFrame, match->bbox));
            cout << "Initialized track from " << referenceImage
                 << " with " << match->matches << " LightGlue matches and bbox "
                 << match->bbox << endl;
        }
    }

    error_code removeError;
    fs::remove(tempFramePath, removeError);

    if (tracks.empty()) {
        cout << "No tracks initialized from the Tracks folder. Select one ROI manually." << endl;
        optional<Track> manualTrack = manualTrackInit(currentFrame);
        if (manualTrack) {
            tracks.push_back(*manualTrack);
        } else {
            cout << "No valid track was initialized" << endl;
            return 0;
        }
    }

    while (true) {
        cap >> currentFrame;
        if (currentFrame.empty()) {
            break;
        }

        for (Track& track : tracks) {
            tracker.updateTracking(currentFrame, track);
            rectangle(currentFrame, track.getDisplayBBox(), Scalar(255, 0, 0), 2);
        }

        imshow("Video Player", currentFrame);

        char c = static_cast<char>(waitKey(25));
        if (c == 27) {
            break;
        }
    }

    return 0;
}

#include "CFT_Track.hpp"
#include "TrackID.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace cv;
using namespace std;


void fft2d(Mat& frame) {
    /*
    computes the 2d Fourier Transform of Frame 
    */
    Mat planes[] = {Mat_<double>(frame), Mat::zeros((frame).size(), CV_64F)};
    merge(planes, 2, frame);

    dft(frame, frame, DFT_COMPLEX_OUTPUT);
}

Mat hanningWindow(int height, int width) {
    /* 
    Creates a hanning window of size width x height
    */
    Mat win;
    createHanningWindow(win, Size(width, height), CV_64F);
    return win;
}

void preProcess(cv::Mat &img, cv::Mat &window) {
    /*
    */
    img.convertTo(img, CV_64F, 1.0/255);
    (img) += 1e-5;
    
    log((img), (img));
    Scalar m, sd;
    cv::meanStdDev((img), m, sd);

    (img) = ((img) - m[0]) / (sd[0] + 0.01);

    mulSpectrums(img, window, img, 0, false);
    return;
}

Mat guass(cv::Mat initFrame, cv::Rect2d rect, int sigma) {
    int width = initFrame.cols;
    int height = initFrame.rows;

    int centerX = rect.x + (0.5 * rect.width);
    int centerY = rect.y + (0.5 * rect.height);

    Mat response = Mat::zeros(height, width, CV_64F);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float distance = (pow((j - centerX), 2) + pow(i - centerY, 2)) / (2 * sigma);
            double ex = exp(-distance);
            response.at<double>(i, j) = ex;
        }
    }
    normalize(response, response, 0, 1, NORM_MINMAX);

    return response;
}

Mat randomWarp(Mat img) {
    /* 
    Adds a random rotation and translation to an img
    */
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> angle_dis(-14.25, 14.25);
    double angle = angle_dis(gen);
    Point2f center(img.cols/2.0F, img.rows/2.0F);
    Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
    Mat warped;
    warpAffine(img, warped, rotMat, img.size(), INTER_LINEAR, BORDER_REFLECT);

    uniform_real_distribution<> trans_dis(-12.0, 12.0);
    float tx = trans_dis(gen), ty = trans_dis(gen);
    Mat warpMat = (Mat_<double>(2,3) << 1, 0, tx, 0, 1, ty);
    warpAffine(warped, warped, warpMat, img.size(), INTER_LINEAR, BORDER_REFLECT);

    return warped;
}

void trainFilter(cv::Mat initFrame, cv::Mat& G, vector<Mat>& A, vector<Mat>& B,
                    int numTrain, double lr) {
    /* 
    trains both the short term and the long term MOSSE filters from the input frame along with 
    the bounding box information for reach corresponding track. 
    */

    int height = (G).rows;
    int width = (G).cols;

    Mat window = hanningWindow(height, width);

    Mat fi = initFrame;

    preProcess(fi, window);
    fft2d(fi);
    
    for(int i = 0; i < 2; i++) {
        mulSpectrums(G, fi, A[i], 0, true);
        mulSpectrums(fi, fi, B[i], 0, true);
    }

    Mat warped(Size(width, height), fi.type()), tmp1(A[0].size(), A[0].type()), tmp2(B[0].size(), B[0].type());
    for (int i = 0; i < numTrain; i++) {
        warped = randomWarp(initFrame);

        preProcess(warped, window);
        fi = warped;
        fft2d(fi);

        mulSpectrums(G, fi, tmp1, 0, true);
        mulSpectrums(fi, fi, tmp2, 0, true);

        for(int i = 0; i < 2; i++) {
            A[i] = (1-lr) * A[i] + (lr) * tmp1;
            B[i] = (1-lr) * B[i] + (lr) * tmp2;
        }
    }
}
    
Track CFT::initTracking(Mat frame) {
    Rect initBBOX = selectROI("Tracker ROI", frame, false, false);
    destroyWindow("Tracker ROI");
    return this->initTracking(frame, initBBOX);
}

Track CFT::initTracking(Mat frame, Rect initBBOX) {

    Track track;
    track.initBBox(frame, initBBOX);
    
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
    grayFrame.convertTo(grayFrame, CV_64F);

    Mat response = guass(grayFrame, track.getSearchArea(), this->sigma);
    Mat window = hanningWindow((response.rows), (response.cols)); 
    //preProcess(response, window);
    
    track.G = track.cropForSearch(response);
    track.fi = track.cropForSearch(grayFrame);

    fft2d(track.G);

    trainFilter(track.fi, track.G, track.A, track.B, this->numTrain, this->lr);

    return track;
}

int CFT::updateTracking(cv::Mat frame, Track &track) {

    // 0 = good track
    // 1 = bad track
    int retVal = 0;

    double maxVal;
    Point maxLoc;

    double psr = 0;
    int dx = 0;
    int dy = 0;

    // change later
    // cout << (track.G.rows) << "::" << (track.G.cols) << endl;
    Mat window = hanningWindow((track.G.rows), (track.G.cols));
    Mat mask = Mat::ones(track.Gi.size(), CV_8U);

    cvtColor(frame, frame, COLOR_BGR2GRAY);

    track.fi = track.cropForSearch(frame);
    preProcess(track.fi, window);
    fft2d(track.fi);
    

    for (int i = 0; i < 2; i++) {

        divSpectrums(track.A[i], track.B[i], track.Hi, 0, false);

        mulSpectrums(track.Hi, track.fi, track.Gi, 0, false);

        idft(track.Gi, track.Gi, DFT_REAL_OUTPUT | DFT_SCALE);

        minMaxLoc(track.Gi, NULL, &maxVal, NULL, &maxLoc);

        Rect peakWindow = Rect(maxLoc.x - 10, maxLoc.y - 10, 21, 21);
        peakWindow &= Rect(0, 0, mask.cols, mask.rows);

        mask(peakWindow) = 0;
        Scalar m, sd;
        meanStdDev(track.Gi, m, sd, mask);
        mask(peakWindow) = 1;

        double tmpPsr = (maxVal - m[0]) / sd[0];
        //cout << "PSR::::::" << tmpPsr << endl;
        if (tmpPsr > psr) {
            psr = tmpPsr;
            dx = int(maxLoc.x - track.Gi.cols / 2);
            dy = int(maxLoc.y - track.Gi.rows / 2);
        }
    }

    track.updateBBox(dx, dy, track.getImageBounds());

    if ((!track.psrFlag) || (psr > 10)) {
        track.fi = track.cropForSearch(frame);
        preProcess(track.fi, window);
        fft2d(track.fi);
    }

    if (!track.psrFlag) {
            track.updateFilter(this->lr, false);
            track.updateFilter(this->lr, true);
            if (psr > 13) track.psrFlag = true;
        }
    else {
        if ((psr > 10) && (psr < 20)) 
            track.updateFilter(this->lr, false);
        else if (psr >= 20) {
            track.updateFilter(this->lr, true);
        }
        else {
            retVal = 1;
            cout << "PSR too low to update" << endl;
        }

    }
    return retVal;
}

//Track CFT::tttf(Mat frame) {
//    // open a select roi window
//    // continue to add real time frames to a buffer in seperate thread
//    //once roi is selected, run the tracker on the buffer until it is sufficiently caught up
//    // free buffer. 
//
//    // object verification?
//    // circular buffer? time limit of 10sec?
//    // transistion between tracking buffer and real time?
//    
//}

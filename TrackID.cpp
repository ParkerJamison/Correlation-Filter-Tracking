#include "TrackID.hpp"
#include <stdexcept>

using namespace cv;
using namespace std;

void Track::initBBox(cv::Mat frame, cv::Rect initBBOX) {

    int rows = frame.rows;
    int cols = frame.cols;
    this->imageBounds = Rect(0, 0, cols, rows);

    this->bbox = initBBOX & this->imageBounds;
    if (this->bbox.width <= 0 || this->bbox.height <= 0) {
        throw runtime_error("Initial bounding box must overlap the frame");
    }
    this->displayBbox = this->bbox;

    int optimalRows = getOptimalDFTSize(this->bbox.height);
    int optimalCols = getOptimalDFTSize(this->bbox.width);

    this->searchArea = this->bbox;

    this->searchArea.x -= (optimalCols - this->bbox.width) / 2;
    this->searchArea.y -= (optimalRows - this->bbox.height) / 2;
    this->searchArea.width = optimalCols;
    this->searchArea.height = optimalRows;

    A.clear();
    B.clear();
    for(int i = 0; i < 2; i++) {
        A.push_back(Mat(optimalRows, optimalCols, CV_64F));
        B.push_back(Mat(optimalRows, optimalCols, CV_64F));
    }

    G.create(optimalRows, optimalCols, CV_64F);
    Gi.create(optimalRows, optimalCols, CV_64F);
    Hi.create(optimalRows, optimalCols, CV_64F);
    fi.create(optimalRows, optimalCols, CV_64F);

}

void Track::updateBBox(int dx, int dy, cv::Rect bounds) {
    
    this->bbox.x += dx;
    this->bbox.y += dy;
    
    this->searchArea.x += dx;
    this->searchArea.y += dy;

    this->displayBbox = this->bbox & this->imageBounds;
}

Mat Track::cropForSearch(Mat frame) {

    Rect tmp = this->searchArea;
    tmp &= this->imageBounds;
    Mat cropped = frame(tmp);


    int left = (0 < -this->searchArea.x) ? -this->searchArea.x : 0;
    int top = (0 < -this->searchArea.y) ? -this->searchArea.y : 0;
    int right = (0 < this->searchArea.x + this->searchArea.width - this->imageBounds.width) 
                    ? this->searchArea.x + this->searchArea.width - this->imageBounds.width : 0;
    int bottom = (0 < this->searchArea.y + this->searchArea.height - this->imageBounds.height) 
                    ? this->searchArea.y + this->searchArea.height - this->imageBounds.height : 0;
    if (top > 0 || bottom > 0 || left > 0 || right > 0) {
        copyMakeBorder(cropped, cropped, top, bottom, left, right, BORDER_REPLICATE);
    }
    return cropped;
}

Mat Track::cropForROI(Mat frame) {
    return frame(this->bbox);
}

void Track::updateFilter(double lr, bool lt) {

    Mat tmp1, tmp2;

    if (fi.size() != G.size()) {
        cout << "RESIZED" << endl;
        resize(fi, fi, G.size());
    }
    mulSpectrums(G, fi, tmp1, 0, true);
    mulSpectrums(fi, fi, tmp2, 0, true);

    if (!lt) {
        A[0] = ((1-lr) * A[0]) + ((0.15) * tmp1);
        B[0] = ((1-lr) * B[0]) + ((0.15) * tmp2);
    }
    else {
        A[1] = ((1-lr) * A[1]) + ((lr) * tmp1);
        B[1] = ((1-lr) * B[1]) + ((lr) * tmp2);
    }
}




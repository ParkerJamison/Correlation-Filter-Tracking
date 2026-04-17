
#ifndef CFT_H
#define CFT_H

#include "opencv2/opencv.hpp"

class Track;
class CFT {
    private:
        double lr;
        int sigma;
        int numTrain;
        cv::Mat window;
        cv::Mat mask;

    public:

        CFT() {
            lr = 0.125;
            sigma = 100;
            numTrain = 16;
        };

        CFT(double x, int y, int z) {
            lr = x;
            sigma = y;
            numTrain = z;
        };

        Track initTracking(cv::Mat frame);
        Track initTracking(cv::Mat frame, cv::Rect initBBOX);
        int updateTracking(cv::Mat frame, Track &track);
};


#endif // CFT_H

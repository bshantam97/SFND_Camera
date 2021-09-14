#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <chrono>

using namespace std;

void detKeypoints1()
{
    // load image from file and convert to grayscale
    cv::Mat imgGray;
    cv::Mat img = cv::imread("../images/img1.png");
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // Shi-Tomasi detector
    int blockSize = 6;       //  size of a block for computing a derivative covariation matrix over each pixel neighborhood/ Gaussian Window Size
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
    double qualityLevel = 0.01;                                   // minimal accepted quality of image corners
    double k = 0.04;
    bool useHarris = false;

    vector<cv::KeyPoint> kptsShiTomasi;
    vector<cv::Point2f> corners;

    // Start the Clock
    auto startTime = std::chrono::steady_clock::now();
    
    // Shi-Tomasi Detector
    cv::goodFeaturesToTrack(imgGray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarris, k);
    
    // End the Clock
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    cout << "Shi-Tomasi with n= " << corners.size() << " keypoints in " << elapsedTime.count() << " ms" << endl;

    for (auto it = corners.begin(); it != corners.end(); ++it)
    { // add corners to result vector

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        kptsShiTomasi.push_back(newKeyPoint);
    }

    // visualize results
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kptsShiTomasi, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Shi-Tomasi Results";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);

    // TODO: use the OpenCV library to add the FAST detector
    // in addition to the already implemented Shi-Tomasi 
    // detector and compare both algorithms with regard to 
    // (a) number of keypoints, (b) distribution of 
    // keypoints over the image and (c) processing speed.
    int threshold = 50;
    bool nms = true;
    std::vector<cv::KeyPoint> keyPoints;

    cv::FastFeatureDetector::DetectorType detectorType = cv::FastFeatureDetector::TYPE_9_16;

    // Create a shared pointer
    cv::Ptr<cv::FastFeatureDetector> create = cv::FastFeatureDetector::create(threshold, nms, detectorType);

    // Start Time
    auto startFAST = std::chrono::steady_clock::now();
    
    // Detect KeyPoints
    create->detect(imgGray, keyPoints);
    
    // End Time
    auto endFAST = std::chrono::steady_clock::now();
    
    // Total Elapsed Time
    auto elapsedFAST = std::chrono::duration_cast<std::chrono::milliseconds>(endFAST-startFAST);
    std::cout << "FAST Algorithm with n = " << keyPoints.size() << " keypoints in " << elapsedFAST.count() << " ms " << std::endl;
    
    cv::Mat visFast = img.clone();
    cv::drawKeypoints(img, keyPoints, visFast, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName1 = "FAST Results";
    cv::namedWindow(windowName1, 1);
    cv::imshow(windowName1, visFast);
    cv::waitKey(0);
 }

int main()
{
    detKeypoints1();
}
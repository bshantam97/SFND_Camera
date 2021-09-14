#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd) Size of Kernel
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix , Kind of like threshold
    double k = 0.04;       // Harris parameter (see equation for details) , Hyperparameters, R = detx - k(trace(x))^2

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1); // 32 Bit floating point single channel
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    // Normalization is used to increase the contrast of the image for better feature extraction
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled); // Converts the image to 8 bit

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    // Need to perform Non Max Suppression to get pixel with maximum cornerness in a local neighborhood
    // and also prevent corners from being too close to each other 
    std::vector<cv::KeyPoint> keyPoints;
    
    // Maximum Permissible Overlap between 2 Keypoints
    // cv::KeyPoints inherits from cv::xfeatures2d::Elliptic_Keypoint
    
    double nmsOverlap = 0.0;
    for (size_t i = 0; i < dst_norm.rows; i++) {
        for (size_t j = 0; j < dst_norm.cols; j++) {
            int cResponse = (int)dst_norm.at<float>(i,j);
            if (cResponse > minResponse) {

                // Only store the Points which are above a particular threshold
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(j,i); // OpenCV stores it as (y,x), stores coordinates
                newKeyPoint.size = 2*apertureSize; // Region around keyPoint used to form description of Keypoint
                newKeyPoint.response = cResponse; // The actual response value

                // Perform NMS around the KeyPoint
                bool overlap = false;
                for (auto it  = keyPoints.begin(); it != keyPoints.end(); ++it) {

                    // This will compute the percent overlap between our keypoint and already stored keypoint
                    double keyPtOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (keyPtOverlap > nmsOverlap) {
                        overlap = true;
                        // If overlap > nmsOverlap and the response is also higher
                        // Replace old Keypoint With new Keypoint
                        if (newKeyPoint.response > (*it).response) {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }

                // Only add new keypoints if no keyPoints have been foind in previous NMS
                if (!overlap) {
                    keyPoints.push_back(newKeyPoint);
                }
            }
        }
    }
    
    std::string windowName2 = "KeyPoints";
    cv::namedWindow(windowName2,5);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keyPoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName2, visImage);
    cv::waitKey(0);
}
int main()
{
    cornernessHarris();
}
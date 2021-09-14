#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <chrono>
#include "structIO.hpp"

// This function takes in an source image, reference image, Keypoints in the source image, Keypoints in the reference image
// Source Descriptors, Reference Descriptor, std::vector<cv::DMatch> basically vector of descriptor matches as reference
// Type of Descriptor (Binary or gradient based), matcher type (Brute force or FLANN) and selectorType (Nearest Neighbor or K-Nearest Neighbor)

void matchDescriptors(cv::Mat &imgSource, cv::Mat &imgRef, std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{

    // configure matcher
    // Cross check mathing which matches the descriptors in both directions
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {

        // what this statement is doing that it checks if my brute force descriptor matcher is taking binary descriptors or gradient based descriptors
        // If the descriptor is binary then we use the Hamming Distance as a similarity measure
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        std::cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        // returns a cv::Ptr<cv::DescriptorMatcher> pointer
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

        //... TODO : implement FLANN matching
        std::cout << "FLANN matching";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        // double t = (double)cv::getTickCount();
        auto startTime = std::chrono::steady_clock::now();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        // t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << " (NN) with n=" << matches.size() << " matches in " << elapsedTime.count() << " ms" << std::endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // TODO : implement k-nearest-neighbor matching
        auto startTime = std::chrono::steady_clock::now();
        std::vector<std::vector<cv::DMatch>> knnMatches;
        
        // Think of it this way that we have been given descriptors in both the images  
        matcher->knnMatch(descSource, descRef, knnMatches, 2);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime);
        std::cout << " (KNN) with n = " << knnMatches.size() << " matches in " << elapsedTime.count() << " ms " << std::endl;
        // TODO : filter matches using descriptor distance ratio test
        const double threshold = 0.8;
        
        for (size_t i = 0; i != knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < threshold * knnMatches[i][1].distance) {
                matches.push_back(knnMatches[i][0]);
            }
        }
        std::cout <<  " Removed matches are " << knnMatches.size() - matches.size() << std::endl;
    }

    // visualize results
    cv::Mat matchImg = imgRef.clone();
    cv::drawMatches(imgSource, kPtsSource, imgRef, kPtsRef, matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::string windowName = "Matching keypoints between two camera images (best 50)";
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    cv::waitKey(0);
}

int main()
{
    cv::Mat imgSource = cv::imread("../images/img1gray.png");
    cv::Mat imgRef = cv::imread("../images/img2gray.png");

    std::vector<cv::KeyPoint> kptsSource, kptsRef; 
    readKeypoints("../dat/C35A5_KptsSource_BRISK_small.dat", kptsSource);
    readKeypoints("../dat/C35A5_KptsRef_BRISK_small.dat", kptsRef);

    cv::Mat descSource, descRef; 
    readDescriptors("../dat/C35A5_DescSource_BRISK_small.dat", descSource);
    readDescriptors("../dat/C35A5_DescRef_BRISK_small.dat", descRef);

    std::vector<cv::DMatch> matches;
    std::string matcherType = "MAT_FLANN"; //MAT_FLANN, MAT_BF
    std::string descriptorType = "DES_BINARY"; 
    std::string selectorType = "SEL_KNN"; //SEL_KNN
    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType);
}
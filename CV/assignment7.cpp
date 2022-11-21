#include "cv.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int main() {
    string targetfilename;
    
    

    Mat query, compare_img, descriptors_query, descriptors_compare;
    Mat bestImg, bestDescriptors;
    vector<KeyPoint> bestKeypoints;
    int matchingCount, mostMatchingCount = 0;
    Ptr<ORB> orbF = ORB::create(1000);
    vector<KeyPoint> keypoints_query, keypoints_compare;
    vector< DMatch > goodMatches;
    BFMatcher matcher(NORM_HAMMING);
    Mat imgMatches;

    int i, k;
    float nndr;

    cout << "Enter query file name: ";
    cin >> targetfilename;
    query = imread("query_image/"+targetfilename);
    if (query.empty()) {
        cout << "no such file!" << endl;
        return -1;
    }
    String path = "DBs/Handong*_1.jpg";
    vector<String> str;
    glob(path, str, false);
    cout << "Sample image load Size: " << str.size() << endl;
    resize(query, query, Size(640, 480));
    imshow("Query", query);
    //Compute ORB Featuresp
    orbF->detectAndCompute(query, noArray(), keypoints_query, descriptors_query);

    for (int cnt = 0; cnt < str.size(); cnt++) {
        compare_img = imread(str[cnt]);
        resize(compare_img, compare_img, Size(640, 480));
        if (compare_img.empty()) return -1;

        orbF->detectAndCompute(compare_img, noArray(), keypoints_compare, descriptors_compare);
        //KNN Matching(k-nearest neighbor matching)
        //Find best and second-best matches
        vector< vector< DMatch> > matches;
        k = 2;
        matcher.knnMatch(descriptors_query, descriptors_compare, matches, k);
        // Find out the best match is definitely better than the second-best match
        nndr = 0.6f;
        matchingCount = 0;
        for (i = 0; i < matches.size(); i++) {
            if (matches.at(i).size() == 2 && matches.at(i).at(0).distance <= nndr * matches.at(i).at(1).distance) {
                matchingCount++;
            }
        }
        cout << "Image Number " << cnt+1 << " Matching: " << matchingCount << endl;

        if (mostMatchingCount < matchingCount) {
            mostMatchingCount = matchingCount;
            bestImg = compare_img;
        }
    }

    vector< vector< DMatch> > matches;

    orbF->detectAndCompute(bestImg, noArray(), keypoints_compare, descriptors_compare);

    //KNN Matching(k-nearest neighbor matching)
    //Find best and second-best matches
    k = 2;
    matcher.knnMatch(descriptors_query, descriptors_compare, matches, k);
    // Find out the best match is definitely better than the second-best match
    nndr = 0.6f;
    for (i = 0; i < matches.size(); i++) {
        if (matches.at(i).size() == 2 &&
            matches.at(i).at(0).distance <= nndr * matches.at(i).at(1).distance) {
            goodMatches.push_back(matches[i][0]);
        }
    }
    //Draws the found matches of keypoints from two images.
    drawMatches(
        query, 
        keypoints_query, 
        bestImg,
        keypoints_compare,
        goodMatches, 
        imgMatches,
        Scalar::all(-1), 
        Scalar(-1), 
        vector<char>(), 
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );
    if (goodMatches.size() < 4) { 
        cout << "Matching failed" << endl; 
        return 0; 
    }
    imshow("Best_matching", imgMatches);
    waitKey(0);

}
#include "cv.hpp"
#include <iostream>

using namespace cv;


int main(int argc, char** argv)
{

    std::string img = "source/lena.png";
    Mat srcImage = imread(img);
    if (!srcImage.data) {
        return 1;
    }
    imshow("srcImage", srcImage);
    waitKey(0);
    return 0;
}
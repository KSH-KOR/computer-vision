#include "cv.hpp"
#include <iostream>

using namespace cv;

class SpatialFiltering {
	Mat getUnsarpMask(Mat mat, int aperture) {
		Mat avg;
		blur(mat, avg, Size(aperture, aperture));
		return mat - avg;
	}
	public:
		Mat sharpeningUsingUnsarpMask(Mat mat, int weight = 2, int aperture = 3) {
			return mat + weight * (this->getUnsarpMask(mat, aperture));
		}
		Mat getHalfRoi(Mat mat, bool isRight = 0) {
			const int width = mat.cols;
			const int height = mat.rows;
			return mat(Rect(isRight ? width / 2 : 0, 0, width / 2, height));
		}
};


int main() {
	Mat moon = imread("moon.png", 0);
	Mat saltandpepper = imread("saltnpepper.png", 0);
	Mat sharpenedImage, medianFilteredImage;

	SpatialFiltering spatialFiltering = SpatialFiltering();

	Mat moonDst = moon.clone();
	Mat matRightHalfROI = spatialFiltering.getHalfRoi(moonDst, true);

	sharpenedImage = spatialFiltering.sharpeningUsingUnsarpMask(matRightHalfROI.clone());
	sharpenedImage.copyTo(matRightHalfROI);

	imshow("moon", moon);
	imshow("moon_filtered", moonDst);

	Mat saltandpepperDst = saltandpepper.clone();
	Mat matLeftHalfROI = spatialFiltering.getHalfRoi(saltandpepperDst, false);

	const int apertureSize = 9;
	medianBlur(matLeftHalfROI, medianFilteredImage, apertureSize);
	medianFilteredImage.copyTo(matLeftHalfROI);

	imshow("saltnpepper", saltandpepper);
	imshow("saltnpepper_filtered", saltandpepperDst);

	waitKey(0);
	return 0;
}


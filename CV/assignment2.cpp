#include "cv.hpp"
#include <iostream>

using namespace cv;

void negativeTransformation(Mat& to, Mat& from, int j, int i) {
	to.at<uchar>(j, i) = 255 - from.at<uchar>(j, i);
}
void gammaTransformation(Mat& to, unsigned char pixelVal, int j, int i) {
	to.at<uchar>(j, i) = pixelVal;
}
Mat rotateBy90Degree(Mat& from) {
	Mat to = from.clone();
	for (int j = 0; j < from.rows; j++) {
		for (int i = 0; i < from.cols; i++) {
			to.at<uchar>(i, to.rows-1-j) = from.at<uchar>(j, i);
		}
	}
	return to;
}

class GammaCorrection {
	unsigned char pix[256];
	float gamma;
	void initPix() {
		for (int i = 0; i < 256; i++) {
			pix[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
		}
	}

	public:
		GammaCorrection(float gamma) {
			this->gamma = gamma;
			initPix();
		}
		unsigned char getGammaCorrectionPixelValue(unsigned char pixVal) {
			return this->pix[pixVal];
		}
		
};

int main() {
	Mat image_gray = imread("source/lena.png", 0); //Read an image ¡°lena.png¡± as a gray-scale image

	float gamma = 10;
	GammaCorrection gammaCorrection = GammaCorrection(gamma);

	Mat rotated_img = rotateBy90Degree(image_gray); //Generate a 90-degree rotated image
	Mat transformated_img = rotated_img.clone();

	for (int j = 0; j < rotated_img.rows; j++) {
		for (int i = 0; i < rotated_img.cols; i++) {
			unsigned char pixVal = rotated_img.at<uchar>(j, i);
			if (pixVal < 127) {
				// perform negative transformation if the pixel value is smaller than 127
				negativeTransformation(transformated_img, rotated_img, j, i);
			}
			else {
				//Otherwise, perform gamma transformation with gamma as 10
				gammaTransformation(transformated_img, gammaCorrection.getGammaCorrectionPixelValue(pixVal), j, i);
			}
		}
	}
			
	imshow("Input image", image_gray); //display 'gray image' for input
	imshow("Result", transformated_img); //display 'result' for result
	waitKey(0);
	return 0;
}

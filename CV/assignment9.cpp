#include "cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const Mat quantization_mat1 = (Mat_<float>(8, 8) <<
	16, 11, 10, 16, 24, 40, 51, 61,
	12, 12, 14, 19, 26, 58, 60, 55,
	14, 13, 16, 24, 40, 57, 69, 56,
	14, 17, 22, 29, 51, 87, 80, 62,
	18, 22, 37, 56, 68, 109, 103, 77,
	24, 35, 55, 64, 81, 104, 113, 92,
	49, 64, 78, 87, 103, 121, 120, 101,
	72, 92, 95, 98, 112, 100, 103, 99
	);
const Mat quantization_mat2 = (Mat_<float>(8, 8) <<
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1
	);
const Mat quantization_mat3 = (Mat_<float>(8, 8) <<
	100, 100, 100, 100, 100, 100, 100, 100,
	100, 100, 100, 100, 100, 100, 100, 100,
	100, 100, 100, 100, 100, 100, 100, 100,
	100, 100, 100, 100, 100, 100, 100, 100,
	100, 100, 100, 100, 100, 100, 100, 100,
	100, 100, 100, 100, 100, 100, 100, 100,
	100, 100, 100, 100, 100, 100, 100, 100,
	100, 100, 100, 100, 100, 100, 100, 100
	);

class ImageCompressor {
	Mat image, image_ycbcr;
	Mat ycbcr_channels[3];
	Mat y_plane;
	Mat dct_src;
	Mat dct_output;
	Mat idct_output;
	Mat result;

	const int imageSize = 512;
	
	

public:
	ImageCompressor(Mat image) {
		this->image = image;
		cvtColor(image, image_ycbcr, CV_BGR2YCrCb);
		split(image_ycbcr, ycbcr_channels);
		y_plane = ycbcr_channels[0];
	}

};

void quantizationing(Mat image, Mat quantization_mat, int isInverse) {
	for (int i = 0; i < image.cols; i++) {
		for (int j = 0; j < image.rows; j++) {
			if (isInverse == 0) {
				image.at<float>(i, j) = roundf(image.at<float>(i, j) / quantization_mat.at<float>(i, j));
			}
			else {
				image.at<float>(i, j) = roundf(image.at<float>(i, j) * quantization_mat.at<float>(i, j));
			}
		}
	}
}

void ShiftYPlane(Mat target, int shift) {
	for (int i = 0; i < 512; i++) {
		for (int j = 0; j < 512; j++) {
			target.at<float>(i, j) += shift;
		}
	}
}

Mat my_dctAndQuantization(Mat src, Mat quantization_mat, int isInverse) {
	const int subImageSize = 8;
	const int cols = src.cols / subImageSize;
	const int rows = src.rows / subImageSize;

	Mat subImage;
	
	if (isInverse == 0) {
		src.convertTo(src, CV_32F);
		ShiftYPlane(src, -128);
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Range rows(subImageSize * i, subImageSize * (i + 1));
			Range cols(subImageSize * j, subImageSize * (j + 1));
			subImage = src(rows, cols);
			if(isInverse == 1) quantizationing(subImage, quantization_mat, isInverse);
			dct(subImage, subImage, isInverse);
			if (isInverse == 0) quantizationing(subImage, quantization_mat, isInverse);
		}
	}
	if (isInverse == 1) {
		ShiftYPlane(src, 128);
		src.convertTo(src, CV_8UC1);
	}
	return src;
}

double getMse(Mat src1, Mat src2) {
	double sum = 0;
	int cols = src1.cols;
	int rows = src1.rows;
	if (cols != src2.cols || rows != src2.rows) return -1.0;
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < cols; j++) {
			sum += pow((src1.at<uchar>(i, j) - src2.at<uchar>(i, j)), 2);
		}
	}
	return sum / (double)(cols * rows);
}

double getPSNR(Mat src1, Mat src2) {
	int maxD = 255;
	return (10 * log10((maxD * maxD) / getMse(src1, src2)));
}

int main() {
	string path = "source/lena.png";
	
	Mat image;
	Mat image_ycbcr;
	Mat ycbcr_channels[3];
	Mat dst(512, 512, CV_32F);
	Mat y(512, 512, CV_8UC1);
	Mat result(512, 512, CV_32F);
	image = imread(path);
	if (image.empty()) {
		return 0;
	}
	cvtColor(image, image_ycbcr, CV_BGR2YCrCb);
	split(image_ycbcr, ycbcr_channels);
	
	for (int j = 0; j < 512; j++) {
		for (int i = 0; i < 512; i++)
		{
			y.at<uchar>(j, i) = 0;
			y.at<uchar>(j, i) = ycbcr_channels[0].at<uchar>(j, i);
		}
	}
	imshow("Original Y", y);
	Mat qm1 = y.clone();
	qm1 = my_dctAndQuantization(qm1, quantization_mat1, 0);
	imshow("QM1 after compression", qm1);
	qm1 = my_dctAndQuantization(qm1, quantization_mat1, 1);
	imshow("QM1 after resolution", qm1);
	cout << "QM1: PSNR = " << getPSNR(y, qm1) << endl;
	Mat qm2 = y.clone();
	qm2 = my_dctAndQuantization(qm2, quantization_mat2, 0);
	imshow("QM2 after compression", qm2);
	qm2 = my_dctAndQuantization(qm2, quantization_mat2, 1);
	imshow("QM2 after resolution", qm2);
	cout << "QM2: PSNR = " << getPSNR(y, qm2) << endl;
	Mat qm3 = y.clone();
	qm3 = my_dctAndQuantization(qm3, quantization_mat3, 0);
	imshow("QM3 after compression", qm3);
	qm3 = my_dctAndQuantization(qm3, quantization_mat3, 1);
	imshow("QM3 after resolution", qm3);
	cout << "QM1: PSNR = " << getPSNR(y, qm3) << endl;

	waitKey(0);
	
	return 0;
}

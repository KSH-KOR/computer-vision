#include "cv.hpp"
#include <iostream>



/*
int main() {
	int width = 150;
	int height = 100;
	cv::Scalar scalar = cv::Scalar(255, 0, 0);
	cv::Mat image = cv::Mat(height, width, CV_8UC(3), scalar);
	std::cout << "mat<" << image.type() << ">" << " information" << std::endl;
	std::cout << "height: " << image.size().height << std::endl;
	std::cout << "width: " << image.size().width << std::endl;

	cv::imshow("image", image);

	cv::waitKey();
	return 0;
}
*/

int main() {
	cv::Mat image = cv::imread("lena.png");
	cv::Size originalImageSize = image.size();
	const int originalImageWidth = originalImageSize.width;
	const int originalImageHeight = originalImageSize.height;
	cv::Mat mask = cv::Mat::zeros(originalImageSize, image.type());
	cv::Mat mask2 = cv::Mat(originalImageSize, image.type(), cv::Scalar(255, 0, 0));
	cv::Mat copied;

	cv::Rect rect = cv::Rect(originalImageWidth/4, originalImageHeight/4, originalImageWidth/2, originalImageHeight/2); // LT position, width, height
	cv::rectangle(mask, rect, cv::Scalar(255, 0, 0), -1, cv::LINE_8, 0);
	imshow("mask", mask);
	image.copyTo(copied, mask2);
	imshow("original", image);
	imshow("copied", copied);
	cv::waitKey(0);
	return 0;
}
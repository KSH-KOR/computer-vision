#include "cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class ImageSegmentation {

	double globalThresholdVal;
	double maxVal;

	Mat src;
	Mat dst;

	void findProperGlobalThresholdVal(int thGiven) {
		int thresh_T  = 200, 
			low_cnt = 0, 
			high_cnt = 0, 
			low_sum = 0, 
			high_sum = 0, 
			th = thGiven;
		int i, j;

		while (1) {
			for (j = 0; j < src.rows; j++) {
				for (i = 0; i < src.cols; i++) {
					if (src.at<uchar>(j, i) < thresh_T) {
						low_sum += src.at<uchar>(j, i);
						low_cnt++;
					}
					else {
						high_sum += src.at<uchar>(j, i);
						high_cnt++;
					}
				}
			}
			if (abs(thresh_T - (low_sum / low_cnt + high_sum / high_cnt) / 2.0f) < th) {
				break;
			}
			else {
				thresh_T = (low_sum / low_cnt + high_sum / high_cnt) / 2.0f;
				low_cnt = high_cnt = low_sum = high_sum = 0;
			}
		}
		globalThresholdVal = thresh_T;

	}

public:
	ImageSegmentation() {

	}
	void setSrc(Mat src) {
		this->src = src;
	}
	void setDst(Mat dst) {
		this->dst = dst;
	}
	Mat globalThreshold(Mat giveSrc, double thresholdVal = 100, double maxVal = 255) {
		setSrc(giveSrc);
		threshold(
			src,
			dst,
			globalThresholdVal == NULL ? thresholdVal : globalThresholdVal,
			maxVal,
			THRESH_BINARY
		);
		return dst;
	}

	Mat adapThreshold(Mat giveSrc, int blockSize = 7, int constantVal = 10) {
		setSrc(giveSrc);
		adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, constantVal);
		return dst;
	}
};

class MyHistogram {

	int hist_h, hist_w, histSize;
	Mat hist;
	Mat normalizedHist;
	Mat histImage;

	Mat getHist(Mat src, const int HISTSIZE) {
		float range[] = { 0, 256 };
		const float* histRange = { range };
		// compute the histograms
		// &src: input image, 1: #of src image, 0: #of channels numerated from 0 ~ channels()-1, Mat(): optional mask
		// hist: output histogram, 1: histogram dimension, &histSize: array of histogram size, &histRange: array of histogram¡¯s boundaries
		calcHist(&src, 1, 0, Mat(), hist, 1, &HISTSIZE, &histRange);
		return hist;
	}

public:
	MyHistogram(Mat src, int hist_h, int hist_w, int histSize) {
		this->hist_h = hist_h;
		this->hist_w = hist_w;
		this->histSize = histSize;
		this->hist = getHist(src, histSize);
		this->histImage = Mat(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	}

	void normalizeHist(int lowerBoundary = 0, int upperBoundary = NULL) {
		normalize(this->hist, this->normalizedHist, lowerBoundary, upperBoundary == NULL ? this->hist_h : upperBoundary, NORM_MINMAX, -1, Mat());
	}
	Mat getHist() {
		return this->hist;
	}
	Mat getNormalizedHist(int lowerBoundary, int upperBoundary) {
		normalizeHist(lowerBoundary, upperBoundary);
		return this->normalizedHist;
	}
	Mat drawHistogram(int marker = -1) {
		Mat histImage;
		// establish the number of bins
		int bin_w = cvRound((double)this->hist_w / this->histSize);
		//draw the histogram
		histImage = Mat(this->hist_h, this->hist_w, CV_8UC3, Scalar(255, 255, 255));
		normalizeHist();
		for (int i = 0; i < this->histSize; i++)
		{
			rectangle(
				histImage, 
				Point(bin_w * i, this->hist_h), 
				Point(bin_w * i + this->hist_w / this->histSize, this->hist_h - cvRound(this->normalizedHist.at<float>(i))), 
				i == marker ? Scalar(0, 0, 255) : Scalar(0, 0, 0), -1
			);
		}
		return histImage;
	}

};


int main() {
	Mat fingerPrint, adaptive1, adaptive;
	Mat fingerPrintAfterThresholding, adaptive1AfterThresholding, adaptiveAfterThresholding;

	const int HIST_H = 512, HIST_W = 512;
	const int HIST_SIZE16 = 16, HIST_SIZE256 = 256;

	fingerPrint = imread("finger_print.png", 0);
	adaptive = imread("adaptive.png", 0);
	adaptive1 = imread("adaptive_1.jpg", 0);

	MyHistogram fingerPrintHistogram = MyHistogram(fingerPrint, HIST_H, HIST_W, HIST_SIZE256);
	Mat hist_graph = fingerPrintHistogram.drawHistogram(165);

	ImageSegmentation imageSegmentation = ImageSegmentation();
	fingerPrintAfterThresholding = imageSegmentation.globalThreshold(fingerPrint, 165);
	adaptiveAfterThresholding = imageSegmentation.adapThreshold(adaptive);
	adaptive1AfterThresholding = imageSegmentation.adapThreshold(adaptive1, 85, 15);

	//imshow("hist", hist_graph);
	imshow("fingerPrint", fingerPrintAfterThresholding);
	imshow("adaptive", adaptiveAfterThresholding);
	imshow("adaptive1", adaptive1AfterThresholding);
	waitKey(0);
	
}
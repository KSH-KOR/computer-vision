#include "cv.hpp"
#include <iostream>

using namespace cv;

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
	Mat drawHistogram() {
		Mat histImage;
		// establish the number of bins
		int bin_w = cvRound((double)this->hist_w / this->histSize);
		//draw the histogram
		histImage = Mat(this->hist_h, this->hist_w, CV_8UC3, Scalar(255, 255, 255));
		normalizeHist();
		for (int i = 0; i < this->histSize; i++)
		{
			rectangle(histImage, Point(bin_w * i, this->hist_h), Point(bin_w * i + this->hist_w / this->histSize, this->hist_h - cvRound(this->normalizedHist.at<float>(i))), Scalar(0, 0, 0), -1);
		}
		return histImage;
	}

};

void myText(Mat paper, int index, float histVal, Point point) {
	double fontScale = 1;
	Scalar color = Scalar(0, 200, 200);
	int thickness = 1;

	putText(paper, format("bin %d : %f", index, histVal), point, FONT_HERSHEY_SIMPLEX, fontScale, color, thickness);
}

void printHistInfoOnMat(Mat paper, Mat hist, int bins, int hist_h) {
	float sum = 0;
	for (int i = 0; i < bins; i++) {
		sum += hist.at<float>(i, 0);
	}
	for (int i = 0; i < bins; i++) {
		myText(paper, i + 1, hist.at<float>(i, 0)/ sum, Point(20, i*30+30));
	}
}


int main() {
	Mat image;
	Mat hist_equalized_image;
	Mat hist_graph;
	Mat hist_equalized_graph;
	Mat hist;
	Mat hist_equalized;

	const int HIST_H = 512, HIST_W = 512;
	const int HIST_SIZE16 = 16, HIST_SIZE8 = 8;

	image = imread("source/moon.png", 0);

	equalizeHist(image, hist_equalized_image);
	MyHistogram histogram16 = MyHistogram(image, HIST_H, HIST_W, HIST_SIZE16);
	MyHistogram histogram16eq = MyHistogram(hist_equalized_image, HIST_H, HIST_W, HIST_SIZE16);

	hist_graph = histogram16.drawHistogram();
	hist_equalized_graph = histogram16eq.drawHistogram();

	MyHistogram histogram8 = MyHistogram(image, HIST_H, HIST_W, HIST_SIZE8);
	MyHistogram histogram8eq = MyHistogram(hist_equalized_image, HIST_H, HIST_W, HIST_SIZE8);

	hist = histogram8.getNormalizedHist(0, HIST_H);
	hist_equalized = histogram8eq.getNormalizedHist(0, HIST_H);

	printHistInfoOnMat(image, hist, HIST_SIZE8, HIST_H);
	printHistInfoOnMat(hist_equalized_image, hist_equalized, HIST_SIZE8, HIST_H);

	imshow("before", image);
	imshow("after", hist_equalized_image);
	imshow("h1", hist_graph);
	imshow("h2", hist_equalized_graph);

	waitKey(0);
	return 0;
}
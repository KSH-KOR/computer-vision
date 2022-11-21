#include "cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

enum MenuOption{ 
	Negative=110, Gamma=103, HistEqualization=104, 
	ColorSlicing=115, ColorConversion=99, AverageFiltering=97,
	WhiteBalancing=119, Reset=114, ESC=27
};

class ImageTransformation {
	unsigned char gammaPix[256];
	float gamma;
	Mat src;
	Mat negativeTransfromated, gammaTransformated;
	uchar pixel;

	void negativeTransformation() {
		negativeTransfromated = src.clone();
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				for (int k = 0; k < 3; k++) {
					negativeTransfromated.at<Vec3b>(i, j)[k] = 255 - src.at<Vec3b>(i, j)[k];
				}
			}
		}
	}
	void gammaTransformation(float gamma) {
		setGamma(gamma);
		gammaTransformated = src.clone();
		for (MatIterator_<Vec3b> it = gammaTransformated.begin<Vec3b>(); it != gammaTransformated.end<Vec3b>(); it++) {
			(*it)[0] = gammaPix[((*it)[0])];
			(*it)[1] = gammaPix[((*it)[1])];
			(*it)[2] = gammaPix[((*it)[2])];
		}
	}
	void initPix() {
		for (int i = 0; i < 256; i++) {
			this->gammaPix[i] = saturate_cast<uchar>(pow((float)(i / 255.0), this->gamma) * 255.0f);
		}
	}
	void setGamma(float gamma) {
		this->gamma = gamma;
		initPix();
	}

public:
	ImageTransformation() {
	}
	void setSrc(Mat src) {
		this->src = src;
	}
	Mat getNegativeTransfromation(Mat src) {
		setSrc(src);
		negativeTransformation();
		return negativeTransfromated;
	}
	Mat getGammaTransformation(Mat src, float gamma = 2.5) {
		setSrc(src);
		gammaTransformation(gamma);
		return gammaTransformated;
	}
};

class ColorProcessing {
	Mat src, HSV, intensity_change, mask_out, change_color, white_balancing, hist_equalizing;
	Mat negativeTransfromated, gammaTransformated;
	int rows, cols;

	void internsityChange() {
		vector<Mat> ic(3);
		split(this->HSV, ic);
		equalizeHist(ic[2], ic[2]);
		merge(ic, this->intensity_change);
		cvtColor(this->intensity_change, this->intensity_change, CV_HSV2BGR);
	}
	void maskOut(int hueLowerBoundary, int hueUpperBoundary) {
		vector<Mat> mo(3);
		uchar* h;
		uchar* s;
		split(HSV, mo);
		for (int j = 0; j < rows; j++) {
			h = mo[0].ptr<uchar>(j);
			s = mo[1].ptr<uchar>(j);
			for (int i = 0; i < cols; i++) {
				if (h[i] > hueLowerBoundary && h[i] < hueUpperBoundary) s[i] = s[i];
				else s[i] = 0;
			}
		}
		merge(mo, mask_out);
		cvtColor(mask_out, mask_out, CV_HSV2BGR);
	}
	void colorChange(int hueIncreasement) {
		vector<Mat> cc(3);
		uchar* h;
		uchar* s;
		split(HSV, cc);
		
		for (int j = 0; j < rows; j++) {
			h = cc[0].ptr<uchar>(j);
			s = cc[1].ptr<uchar>(j);
			for (int i = 0; i < cols; i++) {
				if (h[i] + 50 > 179) h[i] = h[i] + 50 - 179;
				else h[i] += 50;
			}
		}
		merge(cc, change_color);
		cvtColor(change_color, change_color, CV_HSV2BGR);
	}
	void whiteBalacing() {
		Mat bgr_channels[3];
		double avg;
		int sum, temp, i, j, c;

		split(src, bgr_channels);

		for (c = 0; c < src.channels(); c++) {
			sum = 0;
			avg = 0.0f;
			for (i = 0; i < rows; i++) {
				for (j = 0; j < cols; j++) {
					sum += bgr_channels[c].at<uchar>(i, j);
				}
			}
			avg = sum / (rows * cols);

			for (i = 0; i < rows; i++) {
				for (j = 0; j < cols; j++) {
					temp = (128 / avg) * bgr_channels[c].at<uchar>(i, j);
					if (temp > 255) bgr_channels[c].at<uchar>(i, j) = 255;
					else bgr_channels[c].at<uchar>(i, j) = temp;
				}
			}
		}
		white_balancing = src.clone();
		merge(bgr_channels, 3, white_balancing);
	}
	void myEqualizeHist() {
		vector<Mat> channels(3);
		split(HSV, channels);
		equalizeHist(channels[2], channels[2]);
		merge(channels, hist_equalizing);
		cvtColor(hist_equalizing, hist_equalizing, CV_HSV2BGR);
	}
	void negativeTransformation() {
		vector<Mat> nt(3);
		uchar* v;
		split(this->HSV, nt);
		for (int j = 0; j < rows; j++) {
			v = nt[2].ptr<uchar>(j);
			for (int i = 0; i < cols; i++) {
				v[i] = 255 - v[i];
			}
		}
		merge(nt, negativeTransfromated);
		cvtColor(negativeTransfromated, negativeTransfromated, CV_HSV2BGR);
	}

public:
	ColorProcessing() {
	}
	void setSRC(Mat src) {
		this->src = src;
		this->rows = src.rows;
		this->cols = src.cols;
		cvtColor(src, HSV, CV_BGR2HSV);
	}
	Mat getInternsityChange(Mat src) {
		setSRC(src);
		internsityChange();
		return this->intensity_change;
	}
	Mat getMaskOut(Mat src, int hueLowerBoundary = 9, int hueUpperBoundary = 23) {
		setSRC(src);
		maskOut(hueLowerBoundary, hueUpperBoundary);
		return this->mask_out;
	}
	Mat getColorChange(Mat src, int hueIncreasement = 50) {
		setSRC(src);
		colorChange(hueIncreasement);
		return this->change_color;
	}
	Mat getWhiteBalacing(Mat src) {
		setSRC(src);
		whiteBalacing();
		return this->white_balancing;
	}
	Mat getEqualizeHist(Mat src) {
		setSRC(src);
		myEqualizeHist();
		return this->hist_equalizing;
	}
	Mat getNegativeTransfromation(Mat src) {
		setSRC(src);
		negativeTransformation();
		return this->negativeTransfromated;
	}
};

class VideoManager {
	String path;
	VideoCapture cap;
	Mat frame;
	int fps;
	int delay;
	MenuOption currOption = Reset;
	bool isVideoPlaying = true;

	ImageTransformation imageTransfromation = ImageTransformation();
	ColorProcessing colorProcessing = ColorProcessing();

	void setCurrOption(int key) {
		switch (MenuOption(key)) {
			case Negative:
				currOption = Negative;
				break;
			case Gamma:
				currOption = Gamma;
				break;
			case HistEqualization:
				currOption = HistEqualization;
				break;
			case ColorSlicing:
				currOption = ColorSlicing;
				break;
			case ColorConversion:
				currOption = ColorConversion;
				break;
			case AverageFiltering:
				currOption = AverageFiltering;
				break;
			case WhiteBalancing:
				currOption = WhiteBalancing;
				break;
			case Reset:
				currOption = Reset;
				break;
			case ESC:
				isVideoPlaying = false;
				return;
			default:
				currOption = currOption;
				break;
		}
	}

	void getFrame(Mat& currFrame, int key) {
		//currOption = MenuOption(key) == -1 ? currOption : MenuOption(key);
		setCurrOption(key);
		switch (currOption) {
			case Negative:
				currFrame = colorProcessing.getNegativeTransfromation(currFrame);
				break;
			case Gamma:
				currFrame = imageTransfromation.getGammaTransformation(currFrame);
				break;
			case HistEqualization:
				currFrame = colorProcessing.getEqualizeHist(currFrame);
				break;
			case ColorSlicing:
				currFrame = colorProcessing.getMaskOut(currFrame);
				break;
			case ColorConversion:
				currFrame = colorProcessing.getColorChange(currFrame);
				break;
			case AverageFiltering:
				blur(currFrame, currFrame, Size(9, 9));
				break;
			case WhiteBalancing:
				currFrame = colorProcessing.getWhiteBalacing(currFrame);
				break;
			case Reset:
				//currFrame = currFrame;
				break;
			case ESC:
				isVideoPlaying = false;
				break;

		}
	}

public:
	VideoManager(String path) {
		this->path = path;
	}
	void play() {
		if (cap.open(path) == 0) {
			cout << "no such file!" << endl;
			waitKey(0);
		}
		fps = cap.get(CAP_PROP_FPS);
		delay = 1000 / fps; //Find out the proper input parameter for waitKey()

		while (1) {
			cap.read(frame);
			if (frame.empty() || !isVideoPlaying) {
				cout << "end of video" << endl;
				break;
			}
			int key = waitKey(delay);
			getFrame(frame, key);
			imshow("video", frame);
		}
	}
};

int main() {
	VideoManager videoManager = VideoManager("video.mp4");
	videoManager.play();
}
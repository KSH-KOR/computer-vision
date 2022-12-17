#include "cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

enum Status {
	NotDetected, Detected
};

class BackgroundSubtraction {

	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2();
	Mat background, image, gray, result, foregroundMask, foregroundImg;
	Mat sum, avg;
	int cnt = 2;


public:
	BackgroundSubtraction(Mat frame) {
		background = frame.clone();
		resize(background, background, Size(640, 480));
		cvtColor(background, background, CV_BGR2GRAY);
	}
	bool hasDetectedMovingObject(Mat foregroundMask) {
		int count = 0;
		for (int i = 0; i < foregroundMask.rows; i++) {
			for (int j = 0; j < foregroundMask.cols; j++) {
				if (foregroundMask.at<unsigned char>(i, j) == 255) count++;
			}
		}
		
		return (count > 600);
	}

	Mat getForegroundImg() {
		return foregroundImg;
	}
	Mat getForegroundMask() {
		return foregroundMask;
	}
	Mat getBackground() {
		return background;
	}

	void generateAverageImage() {

		if (cnt > 6) {
			add(gray / cnt, background * (cnt - 1) / cnt, background);
			cnt = 0;
		}
		cnt++;
		
	}

	void generateForeground(Mat frame) {
		image = frame;
		cvtColor(image, gray, CV_BGR2GRAY);

		generateAverageImage();
		imshow("back", background);
		absdiff(background, gray, foregroundMask);
		threshold(foregroundMask, foregroundMask, 150, 255, CV_THRESH_BINARY);
		foregroundMask.copyTo(foregroundImg);
		gray.copyTo(foregroundImg, foregroundMask);
	}
};

class VideoManager {
	String path;
	VideoCapture cap;
	Mat frame;
	int fps;
	int delay;
	bool isVideoPlaying = true;
	String alertMsg = "Alert! Moving Object!";
	Point alertMsgLocation = Point(100, 100);

	void printTextonFrame() {
		putText(
			frame,
			alertMsg,
			alertMsgLocation,
			FONT_HERSHEY_SIMPLEX,
			1.5,
			Scalar(255, 255, 255),
			2
		);
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

		Mat background;
		cap >> background;
		BackgroundSubtraction backgroundSubstraction = BackgroundSubtraction(background);
		
		while (1) {
			cap.read(frame);
			if (frame.empty() || !isVideoPlaying) {
				cout << "end of video" << endl;
				break;
			}
			resize(frame, frame, Size(640, 480));
			backgroundSubstraction.generateForeground(frame);
			
			if (backgroundSubstraction.hasDetectedMovingObject(backgroundSubstraction.getForegroundMask())) {
				printTextonFrame();
			}
			imshow("ss", backgroundSubstraction.getForegroundMask());
			imshow("video", frame);
			waitKey(delay);
		}
	}
};

int main() {
	VideoManager videomanager = VideoManager("source/project2-test.mp4");
	videomanager.play();
}
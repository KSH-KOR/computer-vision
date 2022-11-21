#include "cv.hpp"
#include <iostream>


using namespace cv;
using namespace std;

int main() {

	String path = "source/background.mp4";

	Mat frame;
	int fps;
	int delay;
	int currentFrameCount = 0;
	int totalFrameCount;
	VideoCapture cap;
	// check if file exists. if none program ends
	if (cap.open(path) == 0) {
		cout << "no such file!" << endl;
		waitKey(0);
	}

	totalFrameCount = cap.get(CAP_PROP_FRAME_COUNT);
	fps = cap.get(CAP_PROP_FPS);
	delay = 1000 / fps; //Find out the proper input parameter for waitKey()

	
	//Read a video ¡°background.mp4¡± 
	while (1) {
		cap.read(frame);
		if (frame.empty() || currentFrameCount >= fps*3 /*Display video for the first 3 seconds*/) {
			cout << "end of video" << endl;
			break;
		}
		imshow("video", frame);
		waitKey(delay);
		currentFrameCount++;
	}

	// Print out the number of the current frame and the total number of frames
	cout << "the number of the current frame: " << currentFrameCount << endl;
	cout << "the total number of frames: " << totalFrameCount << endl;
}
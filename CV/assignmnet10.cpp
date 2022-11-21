#include "cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

struct MouseParams
{
	Mat img;
	vector<Point2f> out, in;
};
static void onMouse(int event, int x, int y, int, void* param)
{
	MouseParams* mp = (MouseParams*)param;
	Mat img = mp->img;
	
	if (event == EVENT_LBUTTONDOWN) // left button
	{
		//Insert position from LT. Direction is clock-wise
		mp->out.push_back(Point2f(x, y));
	}
	
	//Reset positions
	if (event == EVENT_RBUTTONDOWN)
	{
		mp->out.clear();
	}
}

class VideoManager {
	String fileName1, fileName2, path1, path2;
	String subDirectoryPath;
	VideoCapture cap1, cap2;
	Mat frame1, frame2, result;
	int fps1;
	int delay;
	bool isVideoPlaying = true;
	MouseParams mp;

	void drawCircle() {
		for (size_t i = 0; i < mp.out.size(); i++)
		{
			circle(mp.img, mp.out[i], 3, Scalar(0, 0, 255), 5);
		}
	}
	void getWarpPerspective() {
		Mat homo_mat = getPerspectiveTransform(mp.in, mp.out);
		// apply perspective transformation to img using homo_mat
		// result will have the same size of Size(300, 300) and the same type of img
		warpPerspective(frame2, result, homo_mat, frame2.size());
		Mat mask(frame2.size(), CV_8UC3, Scalar::all(255));
		warpPerspective(mask, mask, homo_mat, mask.size());
		frame1 -= mask;
		frame1 += result;
		//mp.out.clear();
	}

public:
	VideoManager(String fileName1, String fileName2, String subDirectoryPath) {
		this->fileName1 = fileName1;
		this->fileName2 = fileName2;
		this->subDirectoryPath = subDirectoryPath;
	}
	void play() {
		path1 = subDirectoryPath + fileName1 + ".mp4";
		path2 = subDirectoryPath + fileName2 + ".mp4";
		cout << path1;
		if (cap1.open(path1) == 0 || cap2.open(path2) == 0) {
			cout << "no such file!" << endl;
			waitKey(0);
		}
		fps1 = cap1.get(CAP_PROP_FPS);
		delay = 1000 / fps1; //Find out the proper input parameter for waitKey()

		mp.in.push_back(Point2f(0, 0));
		mp.in.push_back(Point2f(640, 0));
		mp.in.push_back(Point2f(640, 480));
		mp.in.push_back(Point2f(0, 480));

		while (1) {
			cap1.read(frame1);
			cap2.read(frame2);
			if ((frame1.empty() && frame2.empty()) || !isVideoPlaying) {
				cout << "end of video" << endl;
				break;
			}
			resize(frame1, frame1, Size(640, 480));
			resize(frame2, frame2, Size(640, 480));

			mp.img = frame1;
			setMouseCallback("Timesquare", onMouse, (void*)&mp);
			
			if (mp.out.size() == 4)
			{
				getWarpPerspective();
			}
			else {
				drawCircle();
			}

			if (!frame1.empty()) imshow(fileName1, frame1);
			if (!frame2.empty()) imshow(fileName2, frame2);
			
			waitKey(delay);
		}
	}
};

int main() {
	String subDirectoryPath = "";
	VideoManager timesquareVideomanager = VideoManager("Timesquare", "contest", subDirectoryPath);
	timesquareVideomanager.play();
}
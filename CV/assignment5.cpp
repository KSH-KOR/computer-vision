#include "cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class LineDetector {
	Mat image, canny_image_left, canny_image_right;
	Mat roi_left, roi_right;
	Mat result;
	vector<Vec2f> lines_left, lines_right;

	Mat getRoi(Mat mat, Point x, Point y) {
		const int width = mat.cols;
		const int height = mat.rows;
		return mat(Rect(x.x, x.y, abs(y.x - x.x), abs(y.y - x.y)));
	}

	void setRoi() {
		assert(!image.empty());
		Mat clone = image.clone();
		roi_left = getRoi(clone, Point(200, 400), Point(600, 600));
		roi_right = getRoi(clone, Point(600, 400), Point(1000, 600));
	}

	void convertColor() {
		assert(!roi_left.empty() && !roi_right.empty());

		cvtColor(roi_left, roi_left, CV_BGR2GRAY);
		cvtColor(roi_right, roi_right, CV_BGR2GRAY);
	}

	void smoothImageUsingGaussianFilter() {
		assert(!roi_left.empty() && !roi_right.empty());
		GaussianBlur(roi_left, roi_left, Size(5, 5), 5, 5, BORDER_DEFAULT);
		GaussianBlur(roi_right, roi_right, Size(5, 5), 5, 5, BORDER_DEFAULT);
	}

	void operateCannyEdge() {
		assert(!roi_left.empty() && !roi_right.empty());
		Canny(roi_left, canny_image_left, 10, 60, 3);
		Canny(roi_right, canny_image_right, 10, 60, 3);
	}

	void getLines(bool isLeftRoi = 1) {
		auto degree30Angle = CV_PI / 6.0;
		double thetaLowerRange = isLeftRoi ? degree30Angle: degree30Angle * 4;
		double thetaUpperRange = thetaLowerRange + degree30Angle;
		Mat target = isLeftRoi ? canny_image_left : canny_image_right;

		
		HoughLines(
			target, 
			isLeftRoi ? lines_left : lines_right, 
			4, 
			CV_PI/180,
			isLeftRoi ? 250 : 150,
			0,
			0,
			thetaLowerRange,
			thetaUpperRange
		);
	}

	void getHoughLines() {
		getLines(true);
		getLines(false);
	}

	void drawLines(Mat image, bool isLeftRoi = 1) {
		vector<Vec2f> lines = isLeftRoi ? lines_left : lines_right;
		float rho_sum = 0, theta_sum = 0;
		float rho_avg, theta_avg, a, b, x0, y0;
		int offset_x1 = isLeftRoi ? 200 : 600;
		int offset_y1 = isLeftRoi ? 400 : 400;
		
		for (auto line : lines) {
				theta_sum += line[1];
				rho_sum += line[0];
		}
		if (theta_sum == 0 || rho_sum == 0) return;

		auto size = lines.size();
		theta_avg = theta_sum / size;
		rho_avg = rho_sum / size;
		a = cos(theta_avg);
		x0 = a * rho_avg;
		b = sin(theta_avg);
		y0 = b * rho_avg;
		auto p1 = Point(cvRound(x0 + 1000 * (-b))+ offset_x1, cvRound(y0 + 1000 * a)+ offset_y1);
		auto p2 = Point(cvRound(x0 - 1000 * (-b))+ offset_x1, cvRound(y0 - 1000 * a)+ offset_y1);
		
		line(image, p1, p2, Scalar(0, 0, 255), 3, 4);
		
	}

	void drawLinesOnTarget() {
		drawLines(image, true);
		drawLines(image, false);
	}

public:
	LineDetector() {
	}

	void displayCannyEdge() {
		assert(!canny_image_left.empty() && !canny_image_right.empty());
		namedWindow("Left canny");
		moveWindow("Left canny", 200, 0);
		imshow("Left canny", canny_image_left);
		namedWindow("Right canny");
		moveWindow("Right canny", 600, 0);
		imshow("Right canny", canny_image_right);
	}


	void process(Mat image) {
		this->image = image;
		setRoi();
		convertColor();
		smoothImageUsingGaussianFilter();
		operateCannyEdge();
		getHoughLines();
		drawLinesOnTarget();
	}

};
class VideoManager {
	String path;
	VideoCapture cap;
	Mat frame;
	int fps;
	int delay;
	int frameCount = 0;

public:
	VideoManager(String path) {
		this->path = path;

	}

	void videoPlay(LineDetector lineDetector) {
		if (cap.open(path) == 0) {
			cout << "no such file!" << endl;
			waitKey(0);
		}
		fps = cap.get(CAP_PROP_FPS);
		frameCount = 0;
		delay = 1000 / fps; //Find out the proper input parameter for waitKey()
		
		while (1) {
			cap.read(frame);
			if (frame.empty() || frameCount >= fps*20) {
				cout << "end of video" << endl;
				break;
			}
			lineDetector.process(frame);
			lineDetector.displayCannyEdge();

			int key = waitKey(delay);
			imshow("Frame", frame);
			frameCount++;

		}
	}
};

int main() {
	VideoManager videoManager = VideoManager("video.mp4");
	videoManager.videoPlay(LineDetector());
}
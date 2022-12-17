#include "cv.hpp"
#include <iostream>
#include <opencv2/dnn.hpp>
#include <fstream>


using namespace cv;
using namespace std;
using namespace dnn;

enum DetectStatus {
	pedestrianDetected, carDetected, startMoving, laneDeparture, none
};

enum ObjectName {
	car, pedestrian, carInfront, unknown
};

enum whichRoi {
	leftSection, rightSection, middleSection
};

class LineDetector {
	Mat image, canny_image_left, canny_image_right, canny_image_middle;
	Mat roi_left, roi_right, roi_middle;
	Mat result;
	vector<Vec2f> lines_left, lines_right, lines_middle;
	double leftXStart, middleXStart, rightXStart;
	double yStart;

	Mat getRoi(Mat mat, Point x, Point y) {
		const int width = mat.cols;
		const int height = mat.rows;
		return mat(Rect(x.x, x.y, abs(y.x - x.x), abs(y.y - x.y)));
	}

	void setRoi() {
		assert(!image.empty());
		Mat clone = image.clone();
		double sectionSizeWidth = clone.size().width / 5;
		double sectionSizeHeight = clone.size().height / 5;

		leftXStart = sectionSizeWidth*1;
		middleXStart = sectionSizeWidth * 2;
		rightXStart = sectionSizeWidth * 3;

		yStart = sectionSizeHeight*4;

		roi_left = getRoi(clone, Point(leftXStart, yStart), Point(leftXStart + sectionSizeWidth, yStart+ sectionSizeHeight));
		roi_middle = getRoi(clone, Point(middleXStart, yStart - sectionSizeHeight), Point(middleXStart + sectionSizeWidth, yStart + sectionSizeHeight));
		roi_right = getRoi(clone, Point(rightXStart, yStart), Point(rightXStart + sectionSizeWidth, yStart+ sectionSizeHeight));
	}

	void convertColor() {
		assert(!roi_left.empty() && !roi_right.empty());

		cvtColor(roi_left, roi_left, CV_BGR2GRAY);
		cvtColor(roi_middle, roi_middle, CV_BGR2GRAY);
		cvtColor(roi_right, roi_right, CV_BGR2GRAY);
	}

	void smoothImageUsingGaussianFilter() {
		assert(!roi_left.empty() && !roi_right.empty());
		GaussianBlur(roi_left, roi_left, Size(5, 5), 5, 5, BORDER_DEFAULT);
		GaussianBlur(roi_middle, roi_middle, Size(5, 5), 5, 5, BORDER_DEFAULT);
		GaussianBlur(roi_right, roi_right, Size(5, 5), 5, 5, BORDER_DEFAULT);
	}

	void operateCannyEdge() {
		assert(!roi_left.empty() && !roi_right.empty());
		Canny(roi_left, canny_image_left, 10, 60, 3);
		Canny(roi_middle, canny_image_middle, 10, 60, 3);
		Canny(roi_right, canny_image_right, 10, 60, 3);
	}

	void getLines(whichRoi whichRoi) {
		auto degree30Angle = CV_PI / 6.0;
		
		double thetaLowerRange = 0; 
		Mat target;
		vector<Vec2f> targetLines;
		int threshold=150;
		switch (whichRoi) {
			case leftSection:
				thetaLowerRange = degree30Angle;
				target = canny_image_left;
				targetLines = lines_left;
				threshold = 150;
				break;
			case rightSection:
				thetaLowerRange = degree30Angle * 4;
				target = canny_image_right;
				targetLines = lines_right;
				threshold = 125;
				break;
			case middleSection:
				thetaLowerRange = -degree30Angle/2;
				target = canny_image_middle;
				targetLines = lines_middle;
				threshold = 150;
				break;
		}
		double thetaUpperRange = thetaLowerRange + degree30Angle;
		HoughLines(
			target,
			targetLines,
			4,
			CV_PI / 180,
			threshold,
			0,
			0,
			thetaLowerRange,
			thetaUpperRange
		);
		switch (whichRoi) {
			case leftSection:
				lines_left = targetLines;
				break;
			case rightSection:
				lines_right = targetLines;
				break;
			case middleSection:
				lines_middle = targetLines;
				break;
		}
	}

	void getHoughLines() {
		getLines(leftSection);
		getLines(middleSection);
		getLines(rightSection);
	}

	void drawLinesOnTarget(Mat image, whichRoi whichRoi){
		vector<Vec2f> lines;
		int offset_x1;
		int offset_y1 = yStart;
		switch (whichRoi) {
		case leftSection:
			lines = lines_left;
			offset_x1 = leftXStart;
			break;
		case rightSection:
			lines = lines_right;
			offset_x1 = rightXStart;
			break;
		case middleSection:
			lines = lines_middle;
			offset_x1 = middleXStart;
			break;
		}
		if (lines.size() == 0) {
			return;
		}
		float rho_sum = 0, theta_sum = 0;
		float rho_avg, theta_avg, a, b, x0, y0;
		

		for (auto line : lines) {
			theta_sum += line[1];
			rho_sum += line[0];
		}

		auto size = lines.size();
		theta_avg = theta_sum / size;
		rho_avg = rho_sum / size;
		a = cos(theta_avg);
		x0 = a * rho_avg;
		b = sin(theta_avg);
		y0 = b * rho_avg;
		auto p1 = Point(cvRound(x0 + 1000 * (-b)) + offset_x1, cvRound(y0 + 1000 * a) + offset_y1);
		auto p2 = Point(cvRound(x0 - 1000 * (-b)) + offset_x1, cvRound(y0 - 1000 * a) + offset_y1);

		line(image, p1, p2, Scalar(0, 0, 255), 1, 4);
	}

	void drawLinesOnTarget() {
		drawLinesOnTarget(image, leftSection);
		drawLinesOnTarget(image, rightSection);
		drawLinesOnTarget(image, middleSection);
	}

public:
	LineDetector() {
	}

	void displayCannyEdge() {
		assert(!canny_image_left.empty() && !canny_image_right.empty());
		namedWindow("Left canny");
		moveWindow("Left canny", leftXStart, 0);
		imshow("Left canny", canny_image_left);
		namedWindow("Right canny");
		moveWindow("Right canny", rightXStart, 0);
		imshow("Right canny", canny_image_right);
	}

	void drawLines() {
		drawLinesOnTarget();
	}
	void process(Mat image) {
		this->image = image;
		setRoi();
		convertColor();
		smoothImageUsingGaussianFilter();
		operateCannyEdge();
		getHoughLines();
	}
	bool isLaneDeparture() {
		return ((lines_left.size() + lines_right.size()) == 0) && (lines_middle.size() > 0);
	}

};

void printAlertMsg(Mat frame, DetectStatus status) {
	const String alertMsgLane = "Lane departure!";
	const String alertMsgCarDetected = "Car detected nearby!";
	const String alertMsgPedestrianDetected = "Human detected nearby!";
	const String alertMsgStartMoving = "Start Moving!";
	const Point alertMsgLocation = Point(100, 100);

	String alertMsg;
	switch (status) {
		case pedestrianDetected:
			alertMsg = alertMsgPedestrianDetected;
			break;
		case carDetected:
			alertMsg = alertMsgCarDetected;
			break;
		case startMoving:
			alertMsg = alertMsgStartMoving;
			break;
		case laneDeparture:
			alertMsg = alertMsgLane;
			break;
		default:
			alertMsg = "";
			break;
	}
	if (status != none) {
		putText(
			frame,
			alertMsg,
			Point(50, 50+status*50),
			FONT_HERSHEY_SIMPLEX,
			1,
			Scalar(0, 0, 255),
			2
		);
	}
	
}

class DetectedObject {
	int id;
	int frameCount = 0;
	int detectedCount = 0;
	int startMovingPrintFrameCount = 0;
	ObjectName objectName;
	DetectStatus status;
	bool isStartMoving = false;
public:
	DetectedObject(int id, ObjectName objectName) {
		this->id = id;
		this->objectName = objectName;
	}
	int getId() {
		return id;
	}
	String getObjectName() {
		switch (objectName) {
			case car:
				return "car";
			case pedestrian:
				return "pedestrian";
			case carInfront:
				return "carInfront";
			case unknown:
				return "unknown";
		}
	}
	int getFrameCount() {
		return frameCount;
	}
	bool shouldErase(Mat frame) {
		bool result = (((frameCount - detectedCount) > 3) || (frameCount > 10));
		bool result1 = (frameCount - detectedCount) > 3;
		if (result1 && objectName == carInfront) {
			isStartMoving = true;
		}
		if (isStartMoving) {
			printAlertMsg(frame, startMoving);
			startMovingPrintFrameCount++;
			if (startMovingPrintFrameCount < 20) {
				return false;
			}
		}
		return result;
	}
	void detected() {
		detectedCount++;
	}
	void alertMsg(Mat frame) {
		switch (objectName) {
			case carInfront:
			case car:
				status = carDetected;
				break;
			case pedestrian:
				status = pedestrianDetected;
				break;
			default:
				status = none;
				break;
		}
		printAlertMsg(frame, status);
		frameCount++;
	}
};

class ObjectDetectManager {
	Mat inputBlob;
	Net net;
	Mat frame;
	const String modelConfiguration = "deep/yolov2-tiny.cfg";
	const String modelBinary = "deep/yolov2-tiny.weights";

	vector<String> classNamesVec;
	Mat detectionMat;

	const float confidenceThreshold = 0.24; //by default
	const int probability_index = 5;

	bool nearbyDetectionAlert = false;

	DetectStatus status = none;
	ObjectName objectName = unknown;

	vector<DetectedObject> detectedObjects = {};
	vector< vector<DetectedObject>::iterator> whatToErase = {};
	vector<DetectedObject>::iterator iter;

	void setNet() {
		net = readNetFromDarknet(modelConfiguration, modelBinary);
		fstream classNamesFile("deep/coco.names");
		if (classNamesFile.is_open()) {
			string className = "";
			while (std::getline(classNamesFile, className)) classNamesVec.push_back(className);
		}
	}

	void getObjectName(String className, bool isItInTheFront = false) {
		if (className == "car") {
			if (isItInTheFront) {
				objectName = carInfront;
				return;
			}
			objectName = car;
			return;
		}
		else if (className == "person") {
			objectName = pedestrian;
			return;
		}
		else {
			objectName = unknown;
			return;
		}
	}

	bool isItInTheFront(int xCenter, int yCenter) {
		int width = frame.size().width;
		int widthSection = width / 5;
		int height = frame.size().height;
		int heightSection = height / 5;
		return widthSection * 2 < xCenter && xCenter < widthSection * 3 && yCenter > heightSection*3.5;
	}

	bool isLaneDeparture() {

	}

	bool doesCarStartMoving() {

	}

	bool isCarDetected(int width, int height) {
		return height > 150 || width > 150;
	}

	bool isPedestrianDetected(int width, int height) {
		return height > 125 || width > 50;
	}

	bool nearbyDetection(int width, int height) {
		switch (objectName) {
		case pedestrian:
			return isPedestrianDetected(width, height);
		case carInfront:
		case car:
			return isCarDetected(width, height);
		default:
			return false;
		}
	}

public:
	ObjectDetectManager() {
		setNet();
	}
	void setTargetFrame(Mat frame) {
		this->frame = frame;
	}
	void generateInputBlobImage() {
		inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
	}
	void generateDetectionMat() {
		net.setInput(inputBlob, "data"); //set the network input
		detectionMat = net.forward("detection_out"); //compute output
	}
	void getLabel() {
		nearbyDetectionAlert = false;
		status = none;
		const int frameCols = frame.cols;
		const int frameRows = frame.rows;
		for (int i = 0; i < detectionMat.rows; i++) {
			const int probability_size = detectionMat.cols - probability_index;
			float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);
			size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
			// prediction probability of each class
			float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
			// for drawing labels with name and confidence
			if (confidence > confidenceThreshold) {
				float x_center = detectionMat.at<float>(i, 0) * frameCols;
				float y_center = detectionMat.at<float>(i, 1) * frameRows;
				float width = detectionMat.at<float>(i, 2) * frameCols;
				float height = detectionMat.at<float>(i, 3) * frameRows;

				Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
				Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
				Rect object(p1, p2);
				Scalar object_roi_color(0, 255, 0);
				rectangle(frame, object, object_roi_color);
				String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] :
					cv::format("unknown(%d)", objectClass);
				getObjectName(className, isItInTheFront(x_center, y_center));
				if (nearbyDetection(width, height)) {
					bool isAdded = false;
					for (iter = detectedObjects.begin(); iter != detectedObjects.end(); iter++) {
						if (i == (*iter).getId()) {
							isAdded = true;
							(*iter).detected();
						}
					}
					if (!isAdded) {
						detectedObjects.push_back(DetectedObject(i, objectName));
					}
				}
				String label = format("id:%d", i);
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				rectangle(frame, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)), object_roi_color, FILLED);
				putText(frame, label, p1 + Point(0, labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
	}
	void printMsg() {
		whatToErase.clear();
		for (vector<DetectedObject>::iterator iter = detectedObjects.begin(); iter != detectedObjects.end(); iter++) {
			cout << "id: " << (*iter).getId() << " object: " << (*iter).getObjectName() << endl;
			if ((*iter).shouldErase(frame)) {
				whatToErase.push_back(iter);
				break;
			}
			(*iter).alertMsg(frame);
		}
		for (auto it : whatToErase) {
			detectedObjects.erase(it);
		}
	}

};

class VideoManager {
	Mat frame;
	String path;
	VideoCapture cap;
	int fps;
	int delay;
	bool isVideoPlaying = true;
	ObjectDetectManager objectDetctor = ObjectDetectManager();
	LineDetector lineDetector = LineDetector();

public:
	VideoManager(String path) {
		this->path = path;
	}

	void play() {
		if (cap.open(path) == 0) {
			cout << "no such file!" << endl;
			waitKey(0);
			return;
		}
		fps = cap.get(CAP_PROP_FPS);
		delay = 1000 / fps; //Find out the proper input parameter for waitKey()
		bool skipping = false;
		int skipFrameCount = 0;
		const int skipAmount = 60;
		int key;
		while (1) {
			cap.read(frame);
			if (frame.empty() || !isVideoPlaying) {
				cout << "end of video" << endl;
				break;
			}
			
			if (skipping) {
				skipFrameCount++;
				if (skipFrameCount > skipAmount) {
					skipFrameCount = 0;
					skipping = false;
				}
				cout << "skipping.." << endl;
				continue;
			}
			resize(frame, frame, Size(640, 480));
			if (frame.channels() == 4) cvtColor(frame, frame, COLOR_BGRA2BGR);

			objectDetctor.setTargetFrame(frame);
			objectDetctor.generateInputBlobImage();
			objectDetctor.generateDetectionMat();
			objectDetctor.getLabel();
			objectDetctor.printMsg();

			lineDetector.process(frame);
			//lineDetector.drawLines();
			//lineDetector.displayCannyEdge();

			if (lineDetector.isLaneDeparture()) {
				printAlertMsg(frame, laneDeparture);
			}

			imshow("Project3", frame);
			key = waitKey(delay);
			if (key == 110) {
				skipping = true;
			}
		}
	}
};

int main() {
	VideoManager videomanager1 = VideoManager("source/Project3_Video/Project3_1.mp4");
	VideoManager videomanager2 = VideoManager("source/Project3_Video/Project3_2.mp4");
	videomanager1.play();
	videomanager2.play();
	return 0;
}
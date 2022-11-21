#include "cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

enum MenuOption {
	NearDetect = 110, FarDetect = 102, MidDetect = 109,
	Tracking = 116, Reset = 114, ESC = 27
};

class Tracker {
	bool _tracking = false;
	bool tracking() {
		return _tracking;
	}
	void tracking(bool newVal) {
		_tracking = newVal;
	}

	void printWarning() {

	}
public:
	void toggleTrackingMode() {
		tracking(!tracking());
	}
	Mat getTracker(Mat currFrame, Rect roiFromUser) {

		Rect m_rc;
		int channels[] = { 0, 1, 2 };
		Mat m_model3d, hsv, m_backproj;
		int hist_sizes[] = { 16, 16, 16 };
		float hrange[] = { 0,180 }; // Hue
		float srange[] = { 0,255 }; // Saturation
		float vrange[] = { 0,255 }; // Brightness
		const float* ranges[] = { hrange, srange, vrange }; // hue, saturation, brightness

		Mat frame = currFrame.clone();
		// convert image from RGB to HSV
		cvtColor(frame, hsv, COLOR_BGR2HSV);
			Rect rc = roiFromUser;
			Mat mask = Mat::zeros(rc.height, rc.width, CV_8U);
			ellipse(mask, Point(rc.width / 2, rc.height / 2), Size(rc.width / 2, rc.height / 2), 0, 0, 360, 255, CV_FILLED);
			Mat roi(hsv, rc);
			//histogram calculation
			calcHist(
				&roi, // The source array(s)
				1, // The number of source arrays
				channels, // The channel (dim) to be measured.
				mask, // A mask to be used on the source array (zeros indicating pixels to be ignored)
				m_model3d, // The Mat object where the histogram will be stored
				3, // The histogram dimensionality.
				hist_sizes, // The number of bins per each used dimension
				ranges // The range of values to be measured per each dimension
			);
			m_rc = rc;
		
		// image processing
			// histogram backprojection.
			// all the arguments are known (the same as used to calculate the histogram), 
			// only we add the backproj matrix, 
			// which will store the backprojection of the source image (&hue)
			calcBackProject(&hsv, 1, channels, m_model3d, m_backproj, ranges);
			// tracking[meanShift]
			// obtain a window with maximum pixel distribution
			meanShift(m_backproj, // dst
				m_rc, // initial location of window
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1) // termination criteria
			);
			Mat result, bgdModel, fgdModel, image, foreground;
			image = currFrame.clone();
			grabCut(image, result, m_rc, bgdModel, fgdModel, 3, GC_INIT_WITH_RECT);
			compare(result, GC_PR_FGD, result, CMP_EQ);
			foreground = Mat(image.size(), image.type(), Scalar(255, 0, 0));
			image.copyTo(foreground, result);
			return foreground;
		
		



	}
	
};

class FaceDetector {
	String faceDetectionConfiguration;
	CascadeClassifier face_classifier;
	Mat grayframe;
	vector<Rect> faces;
	MenuOption option;

	Size getMaxSize() {
		switch (option) {
			case NearDetect:
				return Size(80, 80);
			case MidDetect:
				return Size(56, 56);
			case FarDetect:
				return Size(35, 35);
			
		}
	}
	Size getMinSize() {
		switch (option) {
			case NearDetect:
				return Size(70, 70);
			case MidDetect:
				return Size(55, 55);
			case FarDetect:
				return Size(30, 30);
			
		}
	}
	String getText() {
		switch (option) {
		case NearDetect:
			return "n";
		case MidDetect:
			return "m";
		case FarDetect:
			return "f";

		}
	}

	Rect _roi;

	void setRoi(Point lb, Point tr) {
		this->_roi = Rect(lb, tr);
	}
	

public:
	FaceDetector(String faceDetectionConfiguration) {
		face_classifier.load(faceDetectionConfiguration);
	}
	void setDetectOption(MenuOption option) {
		this->option = option;
	}

	Rect getRoi() {
		return this->_roi;
	}
	
	void detectFrame(Mat frame) {
		cvtColor(frame, grayframe, COLOR_BGR2GRAY);
		face_classifier.detectMultiScale(
			grayframe,
			faces,
			1.1, // increase search scale by 10% each pass
			3, // merge groups of three detections
			0, // not used for a new cascade
			getMinSize(), //minimum size for detection
			getMaxSize()
		);
		// draw the results
		for (int i = 0; i < faces.size(); i++) {
			Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Point tr(faces[i].x, faces[i].y);
			Point msg_location(faces[i].x, faces[i].y + faces[i].height);
			putText(
				frame, 
				getText(), 
				msg_location,
				FONT_HERSHEY_SIMPLEX, 
				1.5,
				Scalar(255, 255, 255), 
				2
			);
			setRoi(lb, tr);
			rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 4, 0);
		}
		
	}

	bool isDeteting() {
		return (option == FarDetect) || (option == MidDetect) || (option == NearDetect);
	}

};

class VideoManager {
	String path;
	VideoCapture cap;
	Mat frame;
	int fps;
	int delay;
	bool isWarning = false;
	bool isTracking = false;
	MenuOption currOption = Reset;
	bool isVideoPlaying = true;
	FaceDetector facedetector = FaceDetector("haarcascade_frontalface_alt.xml");
	Tracker tracker = Tracker();

	void setCurrOption(int key) {
		switch (MenuOption(key)) {
		case NearDetect:
			currOption = NearDetect;
			break;
		case MidDetect:
			currOption = MidDetect;
			break;
		case FarDetect:
			currOption = FarDetect;
			break;
		case ESC:
			isVideoPlaying = false;
			return;
		case Reset:
			currOption = Reset;
			break;
		default:
			currOption = currOption;
			break;
		}
	}

	void getFrame(Mat& currFrame, int key) {
		setCurrOption(key);
		
		switch (currOption) {
		case FarDetect:
		case MidDetect:
		case NearDetect:
			facedetector.setDetectOption(currOption);
			facedetector.detectFrame(currFrame);
			break;
		case ESC:
			isVideoPlaying = false;
			break;
		case Reset:
			facedetector.setDetectOption(Reset);
			currFrame = currFrame;
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
			
			if (facedetector.isDeteting()) {
				isWarning = false;
				if (key == Tracking) {
					if(isTracking) destroyWindow("tracking");
					isTracking = !isTracking;
					
				}
			}
			else {
				if (key == Tracking) {
					isWarning = true;
				}
			}
			if (isWarning) {
				putText(
					frame,
					"Detect before tracking",
					Point(0, 50),
					FONT_HERSHEY_SIMPLEX,
					1.5,
					Scalar(0, 0, 255),
					2
				);
			}
			if (isTracking) {
				imshow("tracking", tracker.getTracker(frame, facedetector.getRoi()));
				
			}
			getFrame(frame, key);
			imshow("video", frame);
			
		}
	}
};

int main() {
	VideoManager videomanager = VideoManager("Faces.mp4");
	videomanager.play();
}
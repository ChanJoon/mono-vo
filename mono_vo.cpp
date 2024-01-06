#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

int getNumSequence(const std::string &seq) {
	if ((seq == "00") || (seq == "02") || (seq == "05") || (seq == "08")) {
		return 4000;
	} else if ((seq == "01") || (seq == "06") || (seq == "07") || (seq == "10")) {
		return 2000;
	}	else if (seq == "03") {
		return 1500;
	}	else if (seq == "04") {
		return 500;
	} else if (seq == "09") {
		return 3000;
	}
	return 0;
}

bool readPoses(const std::string &filename, std::vector<std::vector<double>> &poses) {
	std::ifstream file(filename);
	std::string line;
	if (!file.is_open()) {
		std::cerr << "Failed to open ground truth file: " << filename << std::endl;
		return false;
	}

	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::vector<double> pose;
		double val;

		while (iss >> val) {
			pose.push_back(val);
		}

		if(pose.size() == 12) { // Each line should have 12 values (3x4 transform matrix)
			poses.push_back(pose);
		}
	}

	return true;
}

bool readCalibFile(const std::string &filename, std::vector<double> &output) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open calibration file: " << filename << std::endl;
		return false;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string label;
		iss >> label;
		if (label == "P0:") {
			for (int i = 0; i < 12; ++i){
				iss >> output[i];
			}
			return true;
		}
	}
	return false;
}

int main(int argc, char** argv)
{
	std::string n_seq = "00";
	if (argc >= 2) {
		n_seq = argv[1];
	} else {
		std::cout << "No sequence number provided. Defaulting to sequence 00.\n\n";
	}

// 1. Open KITTI sequences (Images)
	std::cout << "==========  LOAD DATASET   ==========" << std::endl;
	const int MAX_FRAME = getNumSequence(n_seq);
	std::cout << "Max Frame: " << MAX_FRAME << std::endl;


	std::string img_path = "./dataset/sequences/" + n_seq + "/image_0/";
	std::string calib_file = "./dataset/sequences/" + n_seq + "/calib.txt";
	std::string gt_file = "./dataset/poses/" + n_seq + ".txt";

	std::string img_file = img_path + "000000.png";
	cv::Mat img = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cerr << "Could not read the image: " << img_file << std::endl;
		return 1;
	}

	std::vector<std::vector<double>> ground_truth;
	if (readPoses(gt_file, ground_truth)) {
		std::cout << "p0 x: " << ground_truth[0][0] << " y: " << ground_truth[0][1] << " z: " << ground_truth[0][2] << std::endl;
	} else {
		std::cerr << "Could not read ground truth file" << std::endl;
	}

	std::vector<double> projMat(12);
	if (readCalibFile(calib_file, projMat)) {
		std::cout << "Focal Length fx: " << projMat[0] << ", fy: " << projMat[5] << std::endl;
	} else {
		std::cerr << "Could not read calibration file" << std::endl;
	}

	const double focal = projMat[0]; // fx == fy
	const cv::Mat intrMat = (cv::Mat_<double>(3,3) << projMat[0], projMat[1], projMat[2],
																									 projMat[4], projMat[5], projMat[6],
																									 projMat[8], projMat[9], projMat[10]);
	std::cout << "Camera Matrix: \n" << intrMat << std::endl;

// 2. Extract and match features (FAST or etc.)
	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat prevImg, currImg;
	std::vector<cv::Point2f> prevPoints, currPoints;

	currImg = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
	detector->detect(currImg, keypoints);
	cv::KeyPoint::convert(keypoints, prevPoints);

	std::vector<uchar> status;
	std::vector<float> err;
	cv::Size winSize = cv::Size(21, 21);
	const int maxLevel = 3;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);		// maxCount = 30, epsilon = 0.01

// 3. Calculate essential matrix
// (with Nister's five-point algorithm and RANSAC for outlier removal)
	std::cout << "========== SEQUENCE BEGINS ==========" << std::endl;
	auto tic = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < MAX_FRAME; i++) {
		prevImg = currImg.clone();
		currPoints.clear();

		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << i;
		img_file = img_path + ss.str() + ".png";
		currImg = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
		if (currImg.empty()) {
			std::cerr << "Could not read the image: " << currImg << std::endl;
			break;
		}

		// KLT tracker (Locas-Kanade algorithm)
		cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPoints, currPoints, status, err, winSize, maxLevel, criteria);

		cv::Mat E, R, t, mask;
		E = cv::findEssentialMat(currPoints, prevPoints, intrMat, cv::RANSAC, 0.999, 1.0, mask);

// 4. Decompose the matrix into R_k, t_k
		cv::recoverPose(E, currPoints, prevPoints, intrMat, R, t, mask);

// 5. Calculate relative scale and update Coordinates
		// WIP...
	}
	auto toc = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<float>(toc - tic).count();
	std::cout << "Total time: " << duration << " s Avg FPS: " << MAX_FRAME / duration << " fps" << std::endl;




	std::cout << "Success!" << std::endl;
	return 0;
}
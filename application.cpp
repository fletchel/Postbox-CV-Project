#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>


#define NUMBER_OF_POSTBOXES 6
int PostboxLocations[NUMBER_OF_POSTBOXES][8] = {
	{ 26, 113, 106, 113, 13, 133, 107, 134 },
	{ 119, 115, 199, 115, 119, 135, 210, 136 },
	{ 30, 218, 108, 218, 18, 255, 109, 254 },
	{ 119, 217, 194, 217, 118, 253, 207, 253 },
	{ 32, 317, 106, 315, 22, 365, 108, 363 },
	{ 119, 315, 191, 314, 118, 362, 202, 361 } };

#define POSTBOX_TOP_LEFT_COLUMN 0
#define POSTBOX_TOP_LEFT_ROW 1
#define POSTBOX_TOP_RIGHT_COLUMN 2
#define POSTBOX_TOP_RIGHT_ROW 3
#define POSTBOX_BOTTOM_LEFT_COLUMN 4
#define POSTBOX_BOTTOM_LEFT_ROW 5
#define POSTBOX_BOTTOM_RIGHT_COLUMN 6
#define POSTBOX_BOTTOM_RIGHT_ROW 7

#define NUM_FRAMES 96

int PostboxTextLocations[NUMBER_OF_POSTBOXES][2] = {
	{26, 113},
	{119, 115},
	{30, 218},
	{119, 217},
	{32, 317},
	{119, 315}
};


int GroundTruth[95][6]; // first index refers to frame, {-1,...} => obscured, otherwise binary whether post present or not, i.e. {1,0,1,0,0,0} if post present in 1 and 3

using namespace cv;
using namespace std;

int FRAMES_TO_SAVE[] = {8};

// functions used in code in order

//processing functions
Mat getBoxHist(Mat frame, Mat mask, int hbins);
Mat getMask(Mat frame, int points[]);
vector<vector<int>> model(int hbins, float T1, float T2);
void getGroundTruth(int GT[][6]);
vector<float> compare_to_GT(vector<vector<int>> cur_result, int GT[95][6]);

//visualisation functions
void saveHistAsTxt(Mat hist, int hbins, String identifier);
void saveFrame(Mat frame, String identifier);
Mat getAllMasks(Mat frame);
Mat addPostboxText(Mat frame, vector<int> postbox_indices);


/* 

Code divided into two parts, first part is all functions used in actual processing of data

Second part is all functions used to create visualisations etc.*/

/*PART 1: PROCESSING DATA*/

int main()
{
	// get ground truth for use in evaluating performance metrics etc., uncomment this and 
	// use calcMetrics if you want to calculate performance metrics. You will also have to change GroundText.txt path in getGroundTruth().
    
	/*int GT[95][6];
    getGroundTruth(GT);*/

	//evaluate model at these parameters
	vector<vector<int>> cur_model = model(64, 0.9875, 0.85);

	//if we want to check over a range of thresholds, use a for loop and calcMetrics here
	
	return 0;
}



// get histogram of given frame w/ mask

Mat getBoxHist(Mat frame, Mat mask, int hbins) {

	Mat grey_frame;
	
	cvtColor(frame, grey_frame, COLOR_BGR2GRAY);

	

	int histSize[] = { hbins };
	float hranges[] = { 0, 255 };
	const float* ranges[] = { hranges };
	Mat hist;
	int channels[] = { 0 };
	calcHist(&frame, 1, channels, mask,
		hist, 1, histSize, ranges);

	return hist;
}


// used to create masks and get the region of interest, assume corners are in same format as postboxlocations

Mat getMask(Mat frame, int corners[]) {

	int rows = frame.rows;
	int cols = frame.cols;

	Mat mask = Mat(rows, cols, CV_8UC1, Scalar(0));

	vector<Point> region_corners;
	vector<Point> hull;
	region_corners.push_back(Point(corners[0], corners[1]));
	region_corners.push_back(Point(corners[2], corners[3]));
	region_corners.push_back(Point(corners[4], corners[5]));
	region_corners.push_back(Point(corners[6], corners[7]));

	// used to get hull for use in fillConvexPoly
	convexHull(region_corners, hull);

	fillConvexPoly(mask, hull, Scalar(255));


	return mask;

}


// main function for running the model

vector<vector<int>> model(int hbins, float T1, float T2) {

	vector<vector<int>> results;

	VideoCapture postvid("PostboxesWithLines1.avi");

	Mat cur_frame;
	Mat prev_frame;

	Mat baseFrame;
	Mat baseHists[NUMBER_OF_POSTBOXES];

	//get the base frame + hists

	postvid >> baseFrame;

	for (int i = 0; i < NUMBER_OF_POSTBOXES; i++) {
		Mat mask = getMask(baseFrame, PostboxLocations[i]);
		Mat hist = getBoxHist(baseFrame, mask, hbins);

		baseHists[i] = hist;

	}

	// i is number of current frame

	int i = 1;

	// prev_frame used in detecting motion
	prev_frame = baseFrame.clone();

	// manually enter first frame results as our assumption is all boxes are empty in first frame
	// can be changed if not the case in some future clip

	results.push_back({ 0,0,0,0,0,0 });

	while (1) {

		i++;

		postvid >> cur_frame;

		//stop when clip over
		if (!cur_frame.data)
			break;

		// calc hists for use in motion detection
		Mat prev_hist = getBoxHist(prev_frame, Mat(), hbins);
		Mat cur_hist = getBoxHist(cur_frame, Mat(), hbins);

		// result of current frame, binary on whether postboxes have post or not, first index -1 if frame is obscured

		vector<int> cur_result = { 0,0,0,0,0,0 };

		// get correlation of prev_hist and cur_hist

		double compare_motion = compareHist(prev_hist, cur_hist, 0);

		// compare correlation to our threshold T1

		if (compare_motion < T1) {

			cur_result.at(0) = -1;

		}



		// Now we check for post in each postbox
		
		for (int j = 0; j < NUMBER_OF_POSTBOXES; j++) {

			Mat mask = getMask(cur_frame, PostboxLocations[j]);
			Mat hist = getBoxHist(cur_frame, mask, hbins);

			// comparing correlations with base histograms

			double comparison = compareHist(baseHists[j], hist, 0);


			if (comparison < T2 && cur_result.at(0) != -1) {

				cur_result.at(j) = 1;

			}


		}



		// enter current frame results in our 2-d vector of all results

		results.push_back(cur_result);

		prev_frame = cur_frame.clone();

		// display current frame
		if(cur_result.at(0) != -1)
		    addPostboxText(cur_frame, cur_result);

		imshow("Video", cur_frame);
		waitKey(35);


	}

	return results;


}

void getGroundTruth(int GT[][6]) {

	// import and process ground truth txt
	// this is going to be very messy and horrible

	ifstream gtfile;
	gtfile.open("C:\\Users\\Luan\\Desktop\\CVReport Data\\Data\\GroundTruth.txt");
	String l;
	int i = 0;

	while (getline(gtfile, l)) {

		GT[i][0] = 0;
		GT[i][1] = 0;
		GT[i][2] = 0;
		GT[i][3] = 0;
		GT[i][4] = 0;
		GT[i][5] = 0;
	
		if (l.find("View") != -1) {

			GT[i][0] = -1;

		}

		if (l.find("No") != -1){
			;
		
		}

		if(l.find("in 2") != -1) {
		
		    GT[i][1] = 1;
		
		}

		if (l.find("in 1 2") != -1) {

			GT[i][0] = 1;
			GT[i][1] = 1;

		}

		if (l.find("in 1 2 3") != -1) {

			GT[i][0] = 1;
			GT[i][1] = 1;
			GT[i][2] = 1;

		}

		if (l.find("in 1 2 3 4") != -1) {

			GT[i][0] = 1;
			GT[i][1] = 1;
			GT[i][2] = 1;
			GT[i][3] = 1;

		}

		if (l.find("in 1 2 3 4 6") != -1) {

			GT[i][0] = 1;
			GT[i][1] = 1;
			GT[i][2] = 1;
			GT[i][3] = 1;
			GT[i][5] = 1;
		}

		else if (l.find("in 1 2 3 4 5 6") != -1) {

			GT[i][0] = 1;
			GT[i][1] = 1;
			GT[i][2] = 1;
			GT[i][3] = 1;
			GT[i][4] = 1;
			GT[i][5] = 1;
		}

		if (l.find("in 1 2 3 4 5") != -1) {

			GT[i][0] = 1;
			GT[i][1] = 1;
			GT[i][2] = 1;
			GT[i][3] = 1;
			GT[i][4] = 1;
		}

		if (l.find("in 2 3 4 5") != -1) {

			GT[i][1] = 1;
			GT[i][2] = 1;
			GT[i][3] = 1;
			GT[i][4] = 1;
		}

		if (l.find("in 3 4") != -1) {

			GT[i][2] = 1;
			GT[i][3] = 1;
		}

		if (l.find("in 4") != -1) {

			GT[i][3] = 1;
		}	

		i++;
	}

	
 
}

vector<float> compare_to_GT(vector<vector<int>> cur_result, int GT[95][6]) {

	// compare results of our model to the ground truth and return the performance metrics

	int model_present = 0; // num of "post present" in model
	int GT_present = 0; //num of post present in GT
	int agree_present = 0; // intersection of post present in GT and model

	int model_obsc = 0;
	int GT_obsc = 0;
	int agree_obsc = 0;

	for (int i = 0; i < 95; i++) {

		int cur_GT[6];

		if (cur_result.at(i).at(0) == -1)
			model_obsc++;

		if (GT[i][0] == -1)
			GT_obsc++;

		if (cur_result.at(i).at(0) == -1 && GT[i][0] == -1)
			agree_obsc++;

		for (int j = 0; j < 6; j++) {
		    
			if (cur_result.at(i).at(j) == 1)
				model_present++;

			if (GT[i][j] == 1)
				GT_present++;

			if (cur_result.at(i).at(j) == 1 && GT[i][j] == 1)
				agree_present++;
		
		}
	
	}


	float p_present = (float)agree_present / (float)model_present;
	float r_present = (float)agree_present / (float)GT_present;
	float F1_present = 2.0 / ((1.0/p_present) + (1.0/r_present));

	float p_obsc = (float)agree_obsc / (float)model_obsc;
	float r_obsc = (float)agree_obsc / (float)GT_obsc;
	float F1_obsc = 2.0 / ((1.0/p_obsc) + (1.0/r_obsc));


	vector<float> metrics = { p_present, r_present, F1_present, p_obsc, r_obsc, F1_obsc};

	return metrics;


}

/*PART 2: VISUALISING DATA*/

//  function to save data for histogram creation in R etc.

void saveHistAsTxt(Mat hist, int hbins, String identifier) {

	cout << identifier;

	String path_ = "C:\\Users\\Luan\\Desktop\\CVReport Data\\";
	String filename = path_.append(identifier.append(".txt"));

	ofstream histfile;

	histfile.open(filename);

	for (int i = 0; i < hbins; i++) {

		cout << 1;


		histfile << hist.at<float>(i) << endl;
	}

	histfile.close();


}

// function to save frame as image for use in report

void saveFrame(Mat frame, String identifier) {

	String path_ = "C:\\Users\\Luan\\Desktop\\CVReport Data\\figures\\";
	String filename = path_.append(identifier.append(".jpg"));

	imwrite(filename, frame);
}


//function used for one particular figure in report

Mat getAllMasks(Mat frame) {

	int rows = frame.rows;
	int cols = frame.cols;

	Mat masks = Mat(rows, cols, CV_8UC1, Scalar(0));

	for (int i = 0; i < 6; i++) {


		int corners[8] = { 0,0,0,0,0,0,0,0 };
		for (int j = 0; j < 8; j++) {

			corners[j] = PostboxLocations[i][j];

		}
		vector<Point> region_corners = {};
		vector<Point> hull = {};
		region_corners.push_back(Point(corners[0], corners[1]));
		region_corners.push_back(Point(corners[2], corners[3]));
		region_corners.push_back(Point(corners[4], corners[5]));
		region_corners.push_back(Point(corners[6], corners[7]));

		convexHull(region_corners, hull);

		fillConvexPoly(masks, hull, Scalar(255));


	}

	return masks;
}

//add text to frame once processing has completed

Mat addPostboxText(Mat frame, vector<int> postbox_indices) {

	for (int i = 0; i < 6; i++) {

		int x = PostboxTextLocations[i][0];
		int y = PostboxTextLocations[i][1];

		Point curPoint = Point(x, y);

		if (postbox_indices.at(i) == 1) {

			putText(frame, "POST PRESENT", curPoint, FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0));

		}

		else {

			putText(frame, "NO POST", curPoint, FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255));

		}

	}

	return frame;

}

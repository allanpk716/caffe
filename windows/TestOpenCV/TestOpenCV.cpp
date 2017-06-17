#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <vector>
#include <afx.h>
#include "FastSymmetryDetector.h"

#ifdef USE_OPENCV
using namespace caffe; // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

DEFINE_int32(g, 150, "gray_thresh");
DEFINE_int32(h, 100, "hough_vote");
DEFINE_int32(d, 0, "degree");
DEFINE_int32(c, 6, "close");

static Point accumIndex(-1, -1);

cv::Scalar sc_gray = cv::Scalar(160, 160, 160);

cv::Mat rotateImage1(cv::Mat img, int degree)
{
	cv::Mat ucImgRotate;

	double a = sin(degree  * CV_PI / 180);
	double b = cos(degree  * CV_PI / 180);
	int width = img.cols;
	int height = img.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));

	cv::Point center = cv::Point(img.cols / 2, img.rows / 2);

	cv::Mat map_matrix = cv::getRotationMatrix2D(center, degree, 1.0);
	map_matrix.at<double>(0, 2) += (width_rotate - width) / 2;     // 修改坐标偏移
	map_matrix.at<double>(1, 2) += (height_rotate - height) / 2;   // 修改坐标偏移

	cv::warpAffine(img, ucImgRotate, map_matrix, { width_rotate, height_rotate },
		CV_INTER_CUBIC | CV_WARP_FILL_OUTLIERS, cv::BORDER_CONSTANT, sc_gray);

	return ucImgRotate;
}

/**
* Mouse callback, to show the line based on which part of accumulation matrix the cursor is.
*/
static void onMouse(int event, int x, int y, int, void * data) {
	Rect *region = (Rect*)data;
	Point point(x, y);

	if ((*region).contains(point)) {
		accumIndex.x = (point.x - region->x) / 2.0;
		accumIndex.y = (point.y - region->y) * 2.0;
	}
	else {
		accumIndex.x = -1;
		accumIndex.y = -1;
	}
}

void testImage() {

	string filename = "d:\\sample1.jpg";

	namedWindow("");
	moveWindow("", 0, 0);

	Mat frame = imread(filename, -1);

	/* Determine the shape of Hough accumulationmatrix */
	float rho_divs = hypotf(frame.rows, frame.cols) + 1;
	float theta_divs = 180.0;

	FastSymmetryDetector detector(frame.size(), Size(rho_divs, theta_divs), 1);


	Rect region(0, frame.rows, theta_divs * 2.0, rho_divs * 0.5);
	setMouseCallback("", onMouse, static_cast<void*>(&region));
	Mat temp, edge;

	/* Adjustable parameters, depending on the scene condition */
	int canny_thresh_1 = 30;
	int canny_thresh_2 = 90;
	int min_pair_dist = 25;
	int max_pair_dist = 500;
	int no_of_peaks = 1;

	createTrackbar("canny_thresh_1", "", &canny_thresh_1, 500);
	createTrackbar("canny_thresh_2", "", &canny_thresh_2, 500);
	createTrackbar("min_pair_dist", "", &min_pair_dist, 500);
	createTrackbar("max_pair_dist", "", &max_pair_dist, 500);
	createTrackbar("no_of_peaks", "", &no_of_peaks, 10);

	while (true) 
	{
		temp = frame.clone();

		/* Find the edges */
		cvtColor(temp, edge, CV_BGR2GRAY);
		Canny(edge, edge, canny_thresh_1, canny_thresh_2);

		/* Vote for the accumulation matrix */
		detector.vote(edge, min_pair_dist, max_pair_dist);

		/* Draw the symmetrical line */
		vector<pair<Point, Point>> result = detector.getResult(no_of_peaks);
		for (auto point_pair : result)
			line(temp, point_pair.first, point_pair.second, Scalar(0, 0, 255), 2);

		/* Visualize the Hough accum matrix */
		Mat accum = detector.getAccumulationMatrix();
		accum.convertTo(accum, CV_8UC3);
		applyColorMap(accum, accum, COLORMAP_JET);
		resize(accum, accum, Size(), 2.0, 0.5);

		/* Draw lines based on cursor position */
		if (accumIndex.x != -1 && accumIndex.y != -1) {
			pair<Point, Point> point_pair = detector.getLine(accumIndex.y, accumIndex.x);
			line(temp, point_pair.first, point_pair.second, CV_RGB(0, 255, 0), 2);
		}

		/* Show the original and edge images */
		Mat appended = Mat::zeros(temp.rows + accum.rows, temp.cols * 2, CV_8UC3);
		temp.copyTo(Mat(appended, Rect(0, 0, temp.cols, temp.rows)));

		cvtColor(edge, Mat(appended, Rect(temp.cols, 0, edge.cols, edge.rows)), CV_GRAY2BGR);
		accum.copyTo(Mat(appended, Rect(0, temp.rows, accum.cols, accum.rows)));

		imshow("", appended);
		if (waitKey(10) == 'q')
			break;
	}
}

int TestFFT()
{
	//Read a single-channel image
	const char* filename = "d:\\00.jpg";
	Mat srcImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (srcImg.empty())
		return -1;

	namedWindow("source", CV_WINDOW_NORMAL);
	imshow("source", srcImg);
	resizeWindow("source", 100, 100);

//	Mat m_ResImg;
//	GaussianBlur(srcImg, m_ResImg, Size(1, 1), 0, 0);
//
//	namedWindow("m_ResImg", CV_WINDOW_NORMAL);
//	imshow("m_ResImg", m_ResImg);
//	resizeWindow("m_ResImg", 100, 100);
//
//	srcImg = m_ResImg;
//
//	Mat sharpenedLena;
//	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);f
//	filter2D(srcImg, srcImg, srcImg.depth(), kernel);
//
//	namedWindow("sharpenedLena", CV_WINDOW_NORMAL);
//	imshow("sharpenedLena", srcImg);
//	resizeWindow("sharpenedLena", 100, 100);

	Point center(srcImg.cols / 2, srcImg.rows / 2);

	if (FLAGS_d > 0)
	{
		//Rotate source image
		Mat rotMatS = getRotationMatrix2D(center, FLAGS_d, 1.0);
		warpAffine(srcImg, srcImg, rotMatS, srcImg.size(), 1, 0, Scalar(255, 255, 255));

		namedWindow("RotatedSrc", CV_WINDOW_NORMAL);
		imshow("RotatedSrc", srcImg);
		resizeWindow("RotatedSrc", 100, 100);
		//imwrite("imageText_R.jpg",srcImg);
	}

	//Expand image to an optimal size, for faster processing speed
	//Set widths of borders in four directions
	//If borderType==BORDER_CONSTANT, fill the borders with (0,0,0)
	Mat padded;
	int opWidth = getOptimalDFTSize(srcImg.rows);
	int opHeight = getOptimalDFTSize(srcImg.cols);
	copyMakeBorder(srcImg, padded, 0, opWidth - srcImg.rows, 0, opHeight - srcImg.cols, BORDER_CONSTANT, sc_gray);// Scalar::all(0)

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat comImg;
	//Merge into a double-channel image
	merge(planes, 2, comImg);

	//Use the same image as input and output,
	//so that the results can fit in Mat well
	dft(comImg, comImg);

	//Compute the magnitude
	//planes[0]=Re(DFT(I)), planes[1]=Im(DFT(I))
	//magnitude=sqrt(Re^2+Im^2)
	split(comImg, planes);
	magnitude(planes[0], planes[1], planes[0]);

	//Switch to logarithmic scale, for better visual results
	//M2=log(1+M1)
	Mat magMat = planes[0];
	magMat += Scalar::all(1);
	log(magMat, magMat);

	//Crop the spectrum
	//Width and height of magMat should be even, so that they can be divided by 2
	//-2 is 11111110 in binary system, operator & make sure width and height are always even
	magMat = magMat(Rect(0, 0, magMat.cols & -2, magMat.rows & -2));

	//Rearrange the quadrants of Fourier image,
	//so that the origin is at the center of image,
	//and move the high frequency to the corners
	int cx = magMat.cols / 2;
	int cy = magMat.rows / 2;

	Mat q0(magMat, Rect(0, 0, cx, cy));
	Mat q1(magMat, Rect(0, cy, cx, cy));
	Mat q2(magMat, Rect(cx, cy, cx, cy));
	Mat q3(magMat, Rect(cx, 0, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q2.copyTo(q0);
	tmp.copyTo(q2);

	q1.copyTo(tmp);
	q3.copyTo(q1);
	tmp.copyTo(q3);

	//Normalize the magnitude to [0,1], then to[0,255]
	normalize(magMat, magMat, 0, 1, CV_MINMAX);
	Mat magImg(magMat.size(), CV_8UC1);
	magMat.convertTo(magImg, CV_8UC1, 255, 0);

	namedWindow("magnitude", CV_WINDOW_NORMAL);
	imshow("magnitude", magImg);
	resizeWindow("magnitude", 100, 100);
	//imwrite("imageText_mag.jpg",magImg);

	//Turn into binary image
	threshold(magImg, magImg, FLAGS_g, 255, CV_THRESH_BINARY);

	namedWindow("mag_binary", CV_WINDOW_NORMAL);
	imshow("mag_binary", magImg);
	resizeWindow("mag_binary", 100, 100);

	// 闭运算 先膨胀，再腐蚀 Close
	// MORPH_ELLIPSE	MORPH_RECT		MORPH_CROSS
	Mat element_d = getStructuringElement(MORPH_CROSS, Size(FLAGS_c, FLAGS_c));

	dilate(magImg, magImg, element_d);

	namedWindow("Close_d", CV_WINDOW_NORMAL);
	imshow("Close_d", magImg);
	resizeWindow("Close_d", 100, 100);

	erode(magImg, magImg, element_d);

	namedWindow("Close_e", CV_WINDOW_NORMAL);
	imshow("Close_e", magImg);
	resizeWindow("Close_e", 100, 100);
	//imwrite("imageText_bin.jpg",magImg);

	Mat element_d_1 = getStructuringElement(MORPH_CROSS, Size(FLAGS_c - 2, FLAGS_c - 2));

	dilate(magImg, magImg, element_d_1);

	namedWindow("Close_d_1", CV_WINDOW_NORMAL);
	imshow("Close_d_1", magImg);
	resizeWindow("Close_d_1", 100, 100);

	erode(magImg, magImg, element_d_1);

	namedWindow("Close_e_1", CV_WINDOW_NORMAL);
	imshow("Close_e_1", magImg);
	resizeWindow("Close_e_1", 100, 100);

	//------------------------------------------------------------
	// 进行最小外接矩阵计算
	CvBox2D rect;
	rect.angle = 0;
	IplImage IpImgsrc(magImg);
	CvSeq* contour = nullptr;
	CvMemStorage* storage = cvCreateMemStorage(0);
	IplImage* dst_rect = cvCreateImage(cvSize(IpImgsrc.width, IpImgsrc.height), IPL_DEPTH_8U, 3);
	cvCvtColor(&IpImgsrc, dst_rect, CV_GRAY2BGR);
	int iRect_Count = 0;

	cvFindContours(&IpImgsrc, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	for (; contour != nullptr; contour = contour->h_next)
	{
		rect = cvMinAreaRect2(contour, storage);

		cout << "CvBox2D angels: " << rect.angle << endl;

		// 画图
		CvPoint2D32f rect_pts0[4];
		cvBoxPoints(rect, rect_pts0);
		int npts[1];
		npts[0] = 4;
		CvPoint rect_pts[4], *pt = rect_pts;

		for (int i = 0; i<4; i++)
		{
			rect_pts[i] = cvPointFrom32f(rect_pts0[i]);
		}

		cvPolyLine(dst_rect, &pt, npts, 1, 1, CV_RGB(255, 0, 0), 1);

		iRect_Count++;
	}

	namedWindow("rect", CV_WINDOW_NORMAL);
	cvShowImage("rect", dst_rect);
	resizeWindow("rect", 100, 100);

	// 释放资源
	cvReleaseImage(&dst_rect);
	cvReleaseMemStorage(&storage);

	//------------------------------------------------------------
	// 旋转
	Mat dstImg;
	dstImg = rotateImage1(srcImg, rect.angle);

	imwrite("d:\\rotate.jpg", dstImg);

	namedWindow("rotate", CV_WINDOW_NORMAL);
	imshow("rotate", dstImg);
	resizeWindow("rotate", 100, 100);
	//------------------------------------------------------------

	//------------------------------------------------------------
//	//Find lines with Hough Transformation
//	vector<Vec2f> lines;
//	float pi180 = (float)CV_PI / 180;
//	Mat linImg(magImg.size(), CV_8UC3);
//	HoughLines(magImg, lines, 1, pi180, FLAGS_h, 0, 0);
//	int numLines = lines.size();
//	for (int l = 0; l<numLines; l++)
//	{
//		float rho = lines[l][0], theta = lines[l][1];
//		Point pt1, pt2;
//		double a = cos(theta), b = sin(theta);
//		double x0 = a*rho, y0 = b*rho;
//		pt1.x = cvRound(x0 + 1000 * (-b));
//		pt1.y = cvRound(y0 + 1000 * (a));
//		pt2.x = cvRound(x0 - 1000 * (-b));
//		pt2.y = cvRound(y0 - 1000 * (a));
//		line(linImg, pt1, pt2, Scalar(255, 0, 0), 3, 8, 0);
//	}
//
//	namedWindow("lines", CV_WINDOW_NORMAL);
//	imshow("lines", linImg);
//	resizeWindow("lines", 100, 100);
//	//imwrite("imageText_line.jpg",linImg);
//	if (lines.size() == 3) {
//		cout << "found three angels:" << endl;
//		cout << lines[0][1] * 180 / CV_PI << endl << lines[1][1] * 180 / CV_PI << endl << lines[2][1] * 180 / CV_PI << endl << endl;
//	}
//	//------------------------------------------------------------
//
//	//Find the proper angel from the three found angels
//	float angel = 0;
//	float piThresh = (float)CV_PI / 90;
//	float pi2 = CV_PI / 2;
//	for (int l = 0; l<numLines; l++)
//	{
//		float theta = lines[l][1];
//		if (abs(theta) < piThresh || abs(theta - pi2) < piThresh)
//			continue;
//		else {
//			angel = theta;
//			break;
//		}
//	}
//
//	//Calculate the rotation angel
//	//The image has to be square,
//	//so that the rotation angel can be calculate right
//	angel = angel<pi2 ? angel : angel - CV_PI;
//	if (angel != pi2) {
//		float angelT = srcImg.rows*tan(angel) / srcImg.cols;
//		angel = atan(angelT);
//	}
//	float angelD = angel * 180 / (float)CV_PI;
//	cout << "the rotation angel to be applied:" << endl << angelD << endl << endl;
//
//	//Rotate the image to recover
//	Mat rotMat = getRotationMatrix2D(center, angelD, 1.0);
//	Mat dstImg = Mat::ones(srcImg.size(), CV_8UC3);
//	warpAffine(srcImg, dstImg, rotMat, srcImg.size(), 1, 0, sc_gray);
//
//
//	namedWindow("result", CV_WINDOW_NORMAL);
//	imshow("result", dstImg);
//	resizeWindow("result", 100, 100);
	//imwrite("imageText_D.jpg",dstImg);

	waitKey(0);

	return 0;
}

int TestPCA()
{
	testImage();
	return 1;
}

int main(int argc, char** argv)
{
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	return TestFFT();
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif // USE_OPENCV


#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <glog/logging.h>

#include <share.h>
#include <afx.h>

#ifdef USE_OPENCV
using namespace caffe; // NOLINT(build/namespaces)
using std::string;

DEFINE_bool(fillBackGround, false,
	"Fill BackGround");

DEFINE_bool(pipe, false,
	"Use Pipe");

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

cv::Mat rotateImage90(cv::Mat img)
{
	cv::Mat dst;
	cv::Mat dst2;
	cv::transpose(img, dst);
	cv::flip(dst, dst2, 1);
	return dst2;
}

void SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

/* Pair (label, confidence) representing a prediction. */
typedef pair<string, float> Prediction;

typedef pair<int, float> PredictionInt;

typedef pair<string, string> SmallAndBigClass;

class Classifier
{
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	vector<Prediction> Classify(const cv::Mat& img, bool bHaveMeanFile = true, bool bBackGround = false, int N = 5);
	vector<PredictionInt> ClassifyInt(const cv::Mat& img, bool bHaveMeanFile = true, bool bBackGround = false, int N = 5);

	vector<string> labels_;

private:
	void SetMean(const string& mean_file);

	vector<float> Predict(const cv::Mat& img, bool bHaveMeanFile, bool bBackGround);

	void WrapInputLayer(vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		vector<cv::Mat>* input_channels, bool bHaveMeanFile, bool bBackGround);

private:
	boost::shared_ptr<Net<float>> net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file)
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str(), _SH_DENYNO);
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const pair<float, int>& lhs,
	const pair<float, int>& rhs)
{
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static vector<int> Argmax(const vector<float>& v, int N)
{
	vector<pair<float, int>> pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(make_pair(v[i], static_cast<int>(i)));
	partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
vector<Prediction> Classifier::Classify(const cv::Mat& img, bool bHaveMeanFile, bool bBackGround, int N)
{
	vector<float> output = Predict(img, bHaveMeanFile, bBackGround);

	N = std::min<int>(labels_.size(), N);
	vector<int> maxN = Argmax(output, N);
	vector<Prediction> predictions;
	for (int i = 0; i < N; ++i)
	{
		int idx = maxN[i];
		predictions.push_back(make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

/* Return the top N predictions. */
vector<PredictionInt> Classifier::ClassifyInt(const cv::Mat& img, bool bHaveMeanFile, bool bBackGround, int N)
{
	vector<float> output = Predict(img, bHaveMeanFile, bBackGround);

	N = std::min<int>(labels_.size(), N);
	vector<int> maxN = Argmax(output, N);
	vector<PredictionInt> predictions;
	for (int i = 0; i < N; ++i)
	{
		int idx = maxN[i];
		predictions.push_back(make_pair(idx, output[idx]));
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file)
{
	if (mean_file.compare("") == 0)
	{
		return;
	}

	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i)
	{
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

vector<float> Classifier::Predict(const cv::Mat& img, bool bHaveMeanFile, bool bBackGround)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels, bHaveMeanFile, bBackGround);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	vector<cv::Mat>* input_channels, bool bHaveMeanFile, bool bBackGround)
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	if (bBackGround == true)
	{
		// Size Width Height
		double dHeight = sample.rows;
		double dWidth = sample.cols;
		double dWidthAndHeight = 0;

		if (dHeight > dWidth)
		{
			sample = rotateImage90(sample);

			dHeight = sample.rows;
			dWidth = sample.cols;

			dWidthAndHeight = dWidth;
		}
		else if (dWidth > dHeight)
		{
			dWidthAndHeight = dWidth;
		}
		else
		{
			dWidthAndHeight = dWidth;
		}

		// 填充背景色 白色图像
		cv::Mat white_img(cv::Size(dWidthAndHeight, dWidthAndHeight), CV_16UC3, sc_gray);//cv::Scalar::all(255)
																						 // 设置画布绘制区域并复制  
		cv::Rect roi_rect = cv::Rect(0, dWidthAndHeight / 2.0f - dHeight / 2.0f, dWidth, dHeight);
		sample.copyTo(white_img(roi_rect));

		// 然后进行

		sample = white_img;

//		cv::imwrite("d:\\merge.jpg", white_img);
	}

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	if (bHaveMeanFile == true)
	{
		cv::Mat sample_normalized;
		subtract(sample_float, mean_, sample_normalized);

		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		split(sample_normalized, *input_channels);
	}
	else
	{
		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		split(sample_float, *input_channels);
	}

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

vector<string> split(string str, string pattern)
{
	string::size_type pos;
	vector<string> result;
	str += pattern;
	int size = str.size();

	for (int i = 0; i<size; i++)
	{
		pos = str.find(pattern, i);
		if (pos<size)
		{
			string s = str.substr(i, pos - i);
			result.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}

// 得到图像的模糊度 Laplacian
double GetFuzzyValue(cv::Mat imageSource)
{
	cv::Mat imageSobel;

	Laplacian(imageSource, imageSobel, CV_8U);

	//图像的平均灰度  
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];

	return meanValue;
}

//#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" ) // 设置入口地址
int main(int argc, char** argv)
{
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	const bool bFillBackGround = FLAGS_fillBackGround;
	const bool bPipe = FLAGS_pipe;

	if (bPipe == false)
	{
		return 0;
	}

	string model_file;
	string trained_file;
	string mean_file;
	string label_file;

	// 是否需要 Mean File
	bool bHaveMeanFile = true;

	// 赋值所需的启动文件
	model_file = "deploy.prototxt";
	trained_file = "snapshot_iter.caffemodel";
	mean_file = "";
	label_file = "ClassifyInfo.txt";
	bHaveMeanFile = false;

	// 加载模型
	Classifier classifier(model_file, trained_file, mean_file, label_file);

	// 将 ClassifyInfo.txt 中的 大小类 信息读取出来
	vector<SmallAndBigClass> smallAndBigClass_v;
	int iLabels = classifier.labels_.size();
	for (size_t i = 0; i < iLabels; ++i)
	{
		vector<string> v;
		SplitString(classifier.labels_[i], v, " ");

		smallAndBigClass_v.push_back(make_pair(v[1], v[2]));
	}

	std::cout << "----------  Loaded  ----------" << std::endl;

	// 连接管道服务器
	BOOL bRet = WaitNamedPipe(TEXT("\\\\.\\Pipe\\mypipe"), NMPWAIT_WAIT_FOREVER);

	if (bRet == false)
	{
		std::cout <<  "Connect The namedPipe Failed! " << std::endl;
		return 0;
	}

	HANDLE hPipe = CreateFile(            //管道属于一种特殊的文件  
		TEXT("\\\\.\\Pipe\\mypipe"),    //创建的文件名  
		GENERIC_READ | GENERIC_WRITE,   //文件模式  
		0,                              //是否共享  
		NULL,                           //指向一个SECURITY_ATTRIBUTES结构的指针  
		OPEN_EXISTING,                  //创建参数  
		FILE_ATTRIBUTE_NORMAL,          //文件属性(隐藏,只读)NORMAL为默认属性  
		NULL);                          //模板创建文件的句柄

	if (INVALID_HANDLE_VALUE == hPipe)
	{
		std::cout << "Open The Pipe Failed !" << std::endl;
		return 0;
	}

	const int MaxLength = 512;

	while (true)
	{
		char rbuf[MaxLength] = "";
		DWORD rlen = 0;
		// 接受服务发送过来的内容 
		ReadFile(hPipe, rbuf, MaxLength, &rlen, 0);

		// 初始化
		if (strcmp(rbuf, "Init") == 0)
		{
			DWORD wlen = 0;
			char buf[] = "Init#succeed";
			
			if (WriteFile(hPipe, buf, strlen(buf) + 1, &wlen, 0) == FALSE) //向服务器发送内容  
			{
				std::cout << "Write To Pipe Failed !" << std::endl;
				break;
			}
		}
		// 退出程序
		else if (strcmp(rbuf, "GetOut") == 0)
		{
			break;
		}
		// 其他情况
		else
		{
			char *p_const;
			char *info = strtok_s(rbuf, "#", &p_const);

			if (strcmp(info, "Img") != 0)
			{
				continue;
			}

			info = strtok_s(NULL, "#", &p_const);

			CHECK(info) << "Cannot strtok Again" << info;

			cv::Mat img = cv::imread(info, -1);

			CHECK(!img.empty()) << "Unable to decode image " << info;

			//需要检测img是否加载成功，并回应错误
			CString buf;
			buf.Format(_T("Img#%hs"), info);

			vector<PredictionInt> predictions = classifier.ClassifyInt(img, bHaveMeanFile, bFillBackGround);

			CString fuzzy;
			fuzzy.Format(_T("#%f"), GetFuzzyValue(img));
			buf += fuzzy;

			/* Print the top N predictions. */
			for (size_t i = 0; i < predictions.size(); ++i)
			{
				PredictionInt& p = predictions[i];
				CString temp;
				temp.Format(_T("#%d#%hs#%f"), p.first, smallAndBigClass_v[p.first].second.c_str(), p.second);
				buf += temp;
			}

			DWORD len = 0;
			char ttt[MaxLength];
			LPCTSTR p = buf.GetBuffer(0);
			buf.ReleaseBuffer();
			strcpy_s(ttt, buf.GetLength() + 1, CT2CA(p));

			WriteFile(hPipe, ttt, strlen(ttt) + 1, &len, NULL);
		}

		Sleep(50);
	}
	// 关闭管道 
	CloseHandle(hPipe);

	return 1;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif // USE_OPENCV


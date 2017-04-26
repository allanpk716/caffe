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

#ifdef USE_OPENCV
using namespace caffe; // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef pair<string, float> Prediction;

class Classifier
{
public:
	Classifier(const string& model_file,
	           const string& trained_file,
	           const string& mean_file,
	           const string& label_file);

	vector<Prediction> Classify(const cv::Mat& img, bool bHaveMeanFile = true, int N = 5);

private:
	void SetMean(const string& mean_file);

	vector<float> Predict(const cv::Mat& img, bool bHaveMeanFile);

	void WrapInputLayer(vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
	                vector<cv::Mat>* input_channels, bool bHaveMeanFile);

private:
	boost::shared_ptr<Net<float>> net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	vector<string> labels_;
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
	std::ifstream labels(label_file.c_str());
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
vector<Prediction> Classifier::Classify(const cv::Mat& img, bool bHaveMeanFile, int N)
{
	vector<float> output = Predict(img, bHaveMeanFile);

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

vector<float> Classifier::Predict(const cv::Mat& img, bool bHaveMeanFile)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
	                     input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels, bHaveMeanFile);

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
                            vector<cv::Mat>* input_channels, bool bHaveMeanFile)
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

int main(int argc, char** argv)
{
	if (argc != 5 && argc != 6 && argc != 7 && argc != 8)
	{
		std::cerr << "labels.txt : " << "ImageType1 0 \r\n ImageType2 1 \r\n ImageType3 2" << std::endl;
		std::cerr << "TestImgFolderPath : " << "D:\TestImgFolderPath" << std::endl;
		std::cerr << "TestImgLabels.txt : " << "Image1.jpg 0 \r\n Image2.jpg 1 \r\n Image3.jpg 2" << std::endl;

		// 5
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " labels.txt img.jpg" << std::endl;

		std::cerr << " Or " << std::endl;

		// 6
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " mean.binaryproto labels.txt img.jpg" << std::endl;

		std::cerr << " Or " << std::endl;
		//-----------------------------------------------------------------------------------------------------
		// 7
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " labels.txt TestImgFolderPath TestImgLabels.txt ImgOutPut.txt" << std::endl;

		std::cerr << " Or " << std::endl;

		//8
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " mean.binaryproto labels.txt TestImgFolderPath TestImgLabels.txt ImgOutPut.txt" << std::endl;

		return 1;
	}

	google::InitGoogleLogging(argv[0]);

	string model_file;
	string trained_file;
	string mean_file;
	string label_file;

	// 识别一个或者多个图片
	bool bClassifyOneOrMore = true;
	// 是否需要 Mean File
	bool bHaveMeanFile = true;
	
	if (argc == 5 || argc == 7)
	{
		model_file = argv[1];
		trained_file = argv[2];
		mean_file = "";
		label_file = argv[3];
		bHaveMeanFile = false;
	}
	else
	{
		model_file = argv[1];
		trained_file = argv[2];
		mean_file = argv[3];
		label_file = argv[4];
	}

	Classifier classifier(model_file, trained_file, mean_file, label_file);

	// 需要识别多个图片
	if (argc == 7 || argc == 8)
	{
		bClassifyOneOrMore = false;
	}

	if (bClassifyOneOrMore == true)
	{
		string file;
		
		if (argc == 5)
		{
			file = argv[4];
		}
		else if (argc == 6)
		{
			file = argv[5];
		}

		std::cout << "---------- Prediction for "
			<< file << " ----------" << std::endl;

		cv::Mat img = cv::imread(file, -1);
		CHECK(!img.empty()) << "Unable to decode image " << file;
		vector<Prediction> predictions = classifier.Classify(img, bHaveMeanFile);

		/* Print the top N predictions. */
		for (size_t i = 0; i < predictions.size(); ++i)
		{
			Prediction p = predictions[i];
			std::cout << std::fixed << std::setprecision(4)
				<< p.second		// confidence
				<< " - \""
				<< p.first		// labels
				<< "\"" << std::endl;
		}
	}
	else
	{
		string imgfilesPath;		  // D:\PicData\cells\ForTestModel
		string TestImgLabels_file;	 // 0.bmp \r\n 1.bmp
		string outputFile;			// D:\\1.txt

		if (argc == 7)
		{
			imgfilesPath = argv[4];
			TestImgLabels_file = argv[5];
			outputFile = argv[6];
		}
		else if (argc == 8)
		{
			imgfilesPath = argv[5];
			TestImgLabels_file = argv[6];
			outputFile = argv[7];
		}

		std::ifstream TestImglabels(TestImgLabels_file.c_str());
		CHECK(TestImglabels) << "Unable to open TestImglabels file " << TestImgLabels_file;
		string line;
		const char * splitChar = " ";
		int iCount_AllImage = 0;

		int iCount_Confidence_First = 0;
		int iCount_Confidence_First_MoreThan_80 = 0;
		// 识别错误的
		int iCount_Confidence_Error_First_MoreThan_50 = 0;
		int iCount_Confidence_Error_First_MoreThan_60 = 0;
		int iCount_Confidence_Error_First_MoreThan_80 = 0;

		std::cout << "How many pic was prediction : " << std::endl;

		std::ofstream myfile(outputFile, ios::out);

		while (getline(TestImglabels, line))
		{
			// result[0] 图片名称  result[0] 图片对应的认为分类编号对应着 label_file
			vector<string> result = split(line, splitChar);

			string file = imgfilesPath + "\\" + result[0];

			cv::Mat img = cv::imread(file, -1);
			CHECK(!img.empty()) << "Unable to decode image " << file;

			vector<Prediction> predictions = classifier.Classify(img, bHaveMeanFile);
			// 只需要第一个
			Prediction p = predictions[0];
			// p.first --> "c1_NeedTrain 1"
			vector<string> result_prediction = split(p.first, splitChar);

			// 只需要第一个
			Prediction p_2 = predictions[1];
			// p.first --> "c1_NeedTrain 1"
			vector<string> result_prediction_2 = split(p_2.first, splitChar);



			// 需要输出的记录文件
			myfile << result[0] << " " << result_prediction[0]<< " " << result_prediction[1] << " " << result[1] << " " << p.second	
								<< " " << result_prediction_2[0] << " " << result_prediction_2[1] << " " << result[1] << " " << p_2.second
				<< std::endl;

			/*
						0.jpg				w2_NeedTrain						7						7					0.981138
						1.jpg				w2_NeedTrain						7						7					0.999996
						2.jpg				w1_NeedTrain						7						6					0.990527
			*/

			if (result[1].compare(result_prediction[1]) == 0 || result[1][0] == result_prediction[0][0])
			{
				iCount_Confidence_First++;
				if (p.second >= 0.8f)
				{
					iCount_Confidence_First_MoreThan_80++;
				}
			}
			else
			{
				// 这里就是识别错误的
				if (p.second >= 0.8f)
				{
					iCount_Confidence_Error_First_MoreThan_80++;
				}
				else if (p.second >= 0.6f)
				{
					iCount_Confidence_Error_First_MoreThan_60++;
				}
				else if (p.second >= 0.5f)
				{
					iCount_Confidence_Error_First_MoreThan_50++;
				}
			}

			iCount_AllImage++;

			std::cout.width(6);// iCount_AllImage 的输出为 6 位宽  
			std::cout << iCount_AllImage << "."; // 算一个 1
			std::cout << "\b\b\b\b\b\b\b";//回删 6 + 1 个字符，使数字在原地变化
		}

		myfile.close();

		std::cout
			<< "All Test Images :"
			<< iCount_AllImage
			<< std::endl;

		std::cout
			<< "First One Accuracy :"
			<< (float)iCount_Confidence_First / (float)iCount_AllImage
			<< std::endl;

		std::cout
			<< "First One More Than 80 Accuracy :"
			<< (float)iCount_Confidence_First_MoreThan_80 / (float)iCount_AllImage
			<< std::endl;

		std::cout
			<< "First One More Than 80 Accuracy Error :"
			<< (float)iCount_Confidence_Error_First_MoreThan_80 / (float)iCount_AllImage
			<< std::endl;

		std::cout
			<< "First One More Than 60 Accuracy Error :"
			<< (float)iCount_Confidence_Error_First_MoreThan_60 / (float)iCount_AllImage
			<< std::endl;

		std::cout
			<< "First One More Than 50 Accuracy Error :"
			<< (float)iCount_Confidence_Error_First_MoreThan_50 / (float)iCount_AllImage
			<< std::endl;
	}

	return 1;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif // USE_OPENCV


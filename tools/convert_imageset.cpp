// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream> // NOLINT(readability/streams)

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe; // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
	"When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
	"Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
	"The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
	"When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
	"When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
	"Optional: What type should we encode the image as ('png','jpg',...).");

DEFINE_bool(fillBackGround, false,
	"Fill BackGround");

/*
		   p >= 90% ->	0
	50% <= p < 90%	->	1
	30% <= p < 50%	->	2
		   p < 30%	->	3
*/
float ConvertProbability2Index(float inputP)
{
	float fOut = 0;

	if (inputP >= 0.9f)
	{
		fOut = 0;
	}
	else if (inputP >= 0.5f && inputP < 0.9f)
	{
		fOut = 1;
	}
	else if (inputP >= 0.3f && inputP < 0.5f)
	{
		fOut = 2;
	}
	else if (inputP < 0.3f)
	{
		fOut = 3;
	}

	return fOut;
}

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

int ReadCaffeResultFile(const string& filename, std::vector<std::vector<std::string>> &v_total)
{
	const char * splitItem = " ";

	std::ifstream infile(filename);

	std::string line;
	
	int iLines = 0;
	while (std::getline(infile, line))
	{
		std::vector<std::string> v;
		SplitString(line, v, splitItem);

		v_total.push_back(v);

		iLines++;
	}

	return iLines;
}

/*
	bTrainOrTest		这两个文件输入的时候对于人工大分类的索引不一样
	iSmallClassNumber   小分类的个数
	inputModelNumber	输入的模型数，默认需要 Top1 Top 2
	channels			填写 1
*/
bool ReadFloatData2Datum(
	std::vector<std::string> line_1,
	std::vector<std::string> line_2,
	std::vector<std::string> line_3,
	bool bTrainOrTest,
	int iSmallClassNumber, int inputModelNumber, int channels, Datum* datum, std::vector<float *> list_buffer)
{
	datum->set_channels(channels);
	datum->set_height(iSmallClassNumber);
	datum->set_width(inputModelNumber * 2);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);
	int datum_channels = datum->channels();
	int datum_height = datum->height();
	int datum_width = datum->width();
	int datum_size = datum_channels * datum_height * datum_width;
	float * buffer = new float[datum_size];
	// 矩阵默认值为 2
	memset(buffer, 2, sizeof(buffer));

	/*
		一共输入了 3 个模型，每个模型给出了 2 个结果
		也就是一种 6 个输入，每一种输入可能性是 21 种，也可能更多
		也就得到了一个 6 * 21 的矩阵，只有一个通道

		Width = 6
		Height = 21

		caffe_1 对应 0 1 列
		caffe_2 对应 2 3 列
		caffe_2 对应 4 5 列

		小分类 21 个 也就是 Height
	*/

	int iBigClassIndex = 0;

	if (bTrainOrTest == true)
	{
		iBigClassIndex = 9;
	}
	else
	{
		iBigClassIndex = 3;
	}

	// 模型 1 ，在 0 1 列
	int datum_index = 0;

	int Top1_SmallClass_index = atoi(line_1[2].c_str());
	float Top1_Probabilty = atof(line_1[4].c_str()) + 1.5f;
	int Top2_SmallClass_index = atoi(line_1[6].c_str());
	float Top2_Probabilty = atof(line_1[8].c_str()) + 1.5f;

	int BigClass_index = atoi(line_1[iBigClassIndex].c_str());

	// 0 列
	datum_index = Top1_SmallClass_index * datum_width + 0;
	buffer[datum_index] = Top1_Probabilty;
	// 1 列
	datum_index = Top2_SmallClass_index * datum_width + 1;
	buffer[datum_index] = Top2_Probabilty;

	// 模型 2 ，在 2 3 列
	Top1_SmallClass_index = atoi(line_2[2].c_str());
	Top1_Probabilty = atof(line_2[4].c_str()) + 1.5f;
	Top2_SmallClass_index = atoi(line_2[6].c_str());
	Top2_Probabilty = atof(line_2[8].c_str()) + 1.5f;
	// 2 列
	datum_index = Top1_SmallClass_index * datum_width + 2;
	buffer[datum_index] = Top1_Probabilty;
	// 3 列
	datum_index = Top2_SmallClass_index * datum_width + 3;
	buffer[datum_index] = Top2_Probabilty;

	// 模型 3 ，在 4 5 列
	Top1_SmallClass_index = atoi(line_3[2].c_str());
	Top1_Probabilty = atof(line_3[4].c_str());
	Top2_SmallClass_index = atoi(line_3[6].c_str());
	Top2_Probabilty = atof(line_3[8].c_str());
	// 2 列
	datum_index = Top1_SmallClass_index * datum_width + 4;
	buffer[datum_index] = Top1_Probabilty;
	// 3 列
	datum_index = Top2_SmallClass_index * datum_width + 5;
	buffer[datum_index] = Top2_Probabilty;

	datum->set_data(buffer, sizeof(buffer));

	datum->set_label(BigClass_index);

	list_buffer.push_back(buffer);

	return true;
}

/*
	输出一个 6 * 2 的矩阵
*/
bool ReadFloatData2Datum_6_2(
	std::vector<std::string> line_1,
	std::vector<std::string> line_2,
	std::vector<std::string> line_3,
	bool bTrainOrTest,
	int iSmallClassNumber, int inputModelNumber, int channels, Datum* datum, std::vector<float *> list_buffer)
{
	datum->set_channels(channels);
	datum->set_height(iSmallClassNumber);
	datum->set_width(inputModelNumber * 2);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);
	int datum_channels = datum->channels();
	int datum_height = datum->height();
	int datum_width = datum->width();
	int datum_size = datum_channels * datum_height * datum_width;
	float * buffer = new float[datum_size];
	// 矩阵默认值为 2
	memset(buffer, 2, sizeof(buffer));

	/*
	一共输入了 3 个模型，每个模型给出了 2 个结果
	得到了一个 6 * 2 的矩阵

	Width = 6
	Height = 2

	caffe_1 对应 0 1 列
	caffe_2 对应 2 3 列
	caffe_2 对应 4 5 列

	*/

	int iBigClassIndex = 0;

	if (bTrainOrTest == true)
	{
		iBigClassIndex = 9;
	}
	else
	{
		iBigClassIndex = 3;
	}

	// 模型 1 ，在 0 1 列
	int datum_index = 0;

	int Top1_SmallClass_index = atoi(line_1[2].c_str());
	float Top1_Probabilty = atof(line_1[4].c_str()) + 0.5f;
	int Top2_SmallClass_index = atoi(line_1[6].c_str());
	float Top2_Probabilty = atof(line_1[8].c_str()) + 0.5f;

	int BigClass_index = atoi(line_1[iBigClassIndex].c_str());

	// Top 1
	buffer[0] = Top1_SmallClass_index;
	buffer[6] = Top1_Probabilty;
	// Top 2
	buffer[0 + 1] = Top2_SmallClass_index;
	buffer[6 + 1] = Top2_Probabilty;

	// 模型 2 ，在 2 3 列
	Top1_SmallClass_index = atoi(line_2[2].c_str());
	Top1_Probabilty = atof(line_2[4].c_str()) + 0.5f;
	Top2_SmallClass_index = atoi(line_2[6].c_str());
	Top2_Probabilty = atof(line_2[8].c_str()) + 0.5f;
	// Top 1
	buffer[0 + 2] = Top1_SmallClass_index;
	buffer[6 + 2] = Top1_Probabilty;
	// Top 2
	buffer[0 + 3] = Top2_SmallClass_index;
	buffer[6 + 3] = Top2_Probabilty;

	// 模型 3 ，在 4 5 列
	Top1_SmallClass_index = atoi(line_3[2].c_str());
	Top1_Probabilty = atof(line_3[4].c_str()) + 0.5f;
	Top2_SmallClass_index = atoi(line_3[6].c_str());
	Top2_Probabilty = atof(line_3[8].c_str()) + 0.5f;
	// Top 1
	buffer[0 + 4] = Top1_SmallClass_index;
	buffer[6 + 4] = Top1_Probabilty;
	// Top 2
	buffer[0 + 5] = Top2_SmallClass_index;
	buffer[6 + 5] = Top2_Probabilty;

	datum->set_data(buffer, sizeof(buffer));

	datum->set_label(BigClass_index);

	list_buffer.push_back(buffer);

	return true;
}

/*
	输出一个 6 通道 1 * 2
*/
bool ReadFloatData2Datum_6_Channel_1_2(
	std::vector<std::string> line_1,
	std::vector<std::string> line_2,
	std::vector<std::string> line_3,
	bool bTrainOrTest,
	int channels, Datum* datum)
{
	datum->set_channels(channels);
	datum->set_height(1);
	datum->set_width(2);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);

	/*
	caffe_1 对应 0 1 列
	caffe_2 对应 2 3 列
	caffe_2 对应 4 5 列
	*/

	int iBigClassIndex = 0;

	if (bTrainOrTest == true)
	{
		iBigClassIndex = 9;
	}
	else
	{
		iBigClassIndex = 3;
	}

	// 模型 1 ，在 0 1 列
	int datum_index = 0;

	int Top1_SmallClass_index = atoi(line_1[2].c_str());
	float Top1_Probabilty = atof(line_1[4].c_str());
	int Top2_SmallClass_index = atoi(line_1[6].c_str());
	float Top2_Probabilty = atof(line_1[8].c_str());

	int BigClass_index = atoi(line_1[iBigClassIndex].c_str());

	datum->set_label(BigClass_index);

	// Top 1
	datum->add_float_data(Top1_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top1_Probabilty));
	// Top 2
	datum->add_float_data(Top2_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top2_Probabilty));

	// 模型 2 ，在 2 3 列
	Top1_SmallClass_index = atoi(line_2[2].c_str());
	Top1_Probabilty = atof(line_2[4].c_str());
	Top2_SmallClass_index = atoi(line_2[6].c_str());
	Top2_Probabilty = atof(line_2[8].c_str());
	// Top 1
	datum->add_float_data(Top1_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top1_Probabilty));
	// Top 2
	datum->add_float_data(Top2_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top2_Probabilty));


	// 模型 3 ，在 4 5 列
	Top1_SmallClass_index = atoi(line_3[2].c_str());
	Top1_Probabilty = atof(line_3[4].c_str());
	Top2_SmallClass_index = atoi(line_3[6].c_str());
	Top2_Probabilty = atof(line_3[8].c_str());
	// Top 1
	datum->add_float_data(Top1_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top1_Probabilty));
	// Top 2
	datum->add_float_data(Top2_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top2_Probabilty));

	return true;
}

/*
输出一个 12 通道 1 * 1
*/
bool ReadFloatData2Datum_12_Channel_1_1(
	std::vector<std::string> line_1,
	std::vector<std::string> line_2,
	std::vector<std::string> line_3,
	bool bTrainOrTest,
	int channels, Datum* datum)
{
	datum->set_channels(channels);
	datum->set_height(1);
	datum->set_width(1);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);

	/*
	caffe_1 对应 0 1 列
	caffe_2 对应 2 3 列
	caffe_2 对应 4 5 列
	*/

	int iBigClassIndex = 0;

	if (bTrainOrTest == true)
	{
		iBigClassIndex = 9;
	}
	else
	{
		iBigClassIndex = 3;
	}

	// 模型 1 ，在 0 1 列
	int datum_index = 0;

	int Top1_SmallClass_index = atoi(line_1[2].c_str());
	float Top1_Probabilty = atof(line_1[4].c_str());
	int Top2_SmallClass_index = atoi(line_1[6].c_str());
	float Top2_Probabilty = atof(line_1[8].c_str());

	int BigClass_index = atoi(line_1[iBigClassIndex].c_str());

	datum->set_label(BigClass_index);

	// Top 1
	datum->add_float_data(Top1_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top1_Probabilty));
	// Top 2
	datum->add_float_data(Top2_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top2_Probabilty));

	// 模型 2 ，在 2 3 列
	Top1_SmallClass_index = atoi(line_2[2].c_str());
	Top1_Probabilty = atof(line_2[4].c_str());
	Top2_SmallClass_index = atoi(line_2[6].c_str());
	Top2_Probabilty = atof(line_2[8].c_str());
	// Top 1
	datum->add_float_data(Top1_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top1_Probabilty));
	// Top 2
	datum->add_float_data(Top2_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top2_Probabilty));


	// 模型 3 ，在 4 5 列
	Top1_SmallClass_index = atoi(line_3[2].c_str());
	Top1_Probabilty = atof(line_3[4].c_str());
	Top2_SmallClass_index = atoi(line_3[6].c_str());
	Top2_Probabilty = atof(line_3[8].c_str());
	// Top 1
	datum->add_float_data(Top1_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top1_Probabilty));
	// Top 2
	datum->add_float_data(Top2_SmallClass_index);
	datum->add_float_data(ConvertProbability2Index(Top2_Probabilty));

	return true;
}


int main(int argc, char** argv)
{
#ifdef USE_OPENCV
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
		"format used as input for Caffe.\n"
		"Usage:\n"
		"    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
		"The ImageNet dataset for the training demo is at\n"
		"    http://www.image-net.org/download-images\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 4)
	{
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
		return 1;
	}

	bool bXoR = false;
	bool bExtend = false;
	bool bTrainOrTest = false;

	string filename_1;
	string filename_2;
	string filename_3;

	int iLines_1 = 0;
	int iLines_2 = 0;
	int iLines_3 = 0;

	std::vector<std::vector<std::string>> v_total_1;
	std::vector<std::vector<std::string>> v_total_2;
	std::vector<std::vector<std::string>> v_total_3;

	// 扩展本程序的使用，需要在后面额外加入三个 训练 结果文件
	// 7 个就是 Test
	if (argc == 7)
	{
		bExtend = true;
		bTrainOrTest = false;
		iLines_1 = ReadCaffeResultFile(argv[4], v_total_1);
		iLines_2 = ReadCaffeResultFile(argv[5], v_total_2);
		iLines_3 = ReadCaffeResultFile(argv[6], v_total_3);
	}
	else if (argc == 8)
	{
		bExtend = true;
		bTrainOrTest = true;
		iLines_1 = ReadCaffeResultFile(argv[4], v_total_1);
		iLines_2 = ReadCaffeResultFile(argv[5], v_total_2);
		iLines_3 = ReadCaffeResultFile(argv[6], v_total_3);
	}
	else if (argc == 9)
	{
		bExtend = true;
		bXoR = true;
	}

	if (iLines_1 != iLines_2 || iLines_2 != iLines_3)
	{
		LOG(INFO) << "3 file lines not the same!";
		return 1;
	}

	const bool is_color = !FLAGS_gray;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;
	const bool bFillBackGround = FLAGS_fillBackGround;

	std::ifstream infile(argv[2]);
	std::vector<std::pair<std::string, int>> lines;
	std::string line;
	size_t pos;
	int label;
	while (std::getline(infile, line))
	{
		pos = line.find_last_of(' ');
		label = atoi(line.substr(pos + 1).c_str());
		lines.push_back(std::make_pair(line.substr(0, pos), label));
	}

	// 是否打乱顺序
	if (FLAGS_shuffle)
	{
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		shuffle(lines.begin(), lines.end());
	}

	LOG(INFO) << "A total of " << lines.size() << " images.";

	if (encode_type.size() && !encoded)
		LOG(INFO) << "encode_type specified, assuming encoded=true.";

	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);

	// Create new DB
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[3], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Storing to db
	std::string root_folder(argv[1]);
	Datum datum;
	std::vector<float *> list_buffer;
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;

	int iCount = 0, iCountTrain = 0, iCountTest = 0;

	if (bExtend == false)
	{
		for (int line_id = 0; line_id < lines.size(); ++line_id)
		{
			bool status;
			std::string enc = encode_type;
			if (encoded && !enc.size())
			{
				// Guess the encoding type from the file name
				string fn = lines[line_id].first;
				size_t p = fn.rfind('.');
				if (p == fn.npos)
					LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
				enc = fn.substr(p);
				std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
			}
			status = ReadImageToDatum(root_folder + lines[line_id].first,
				lines[line_id].second, resize_height, resize_width, is_color,
				enc, &datum);
			if (status == false) continue;
			if (check_size)
			{
				if (!data_size_initialized)
				{
					data_size = datum.channels() * datum.height() * datum.width();
					data_size_initialized = true;
				}
				else
				{
					const std::string& data = datum.data();
					CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
						<< data.size();
				}
			}
			// sequential
			string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

			// Put in db
			string out;
			CHECK(datum.SerializeToString(&out));
			txn->Put(key_str, out);

			if (++count % 1000 == 0)
			{
				// Commit db
				txn->Commit();
				txn.reset(db->NewTransaction());
				LOG(INFO) << "Processed " << count << " files.";
			}
		} // end for
	}
	else
	{
		if (bXoR == true)
		{
			// seed random generator
			std::srand(std::time(NULL));
			const int iNumberOfSamples = std::atoi(argv[7]);

			// generate random data
			typedef int tInput;
			typedef std::pair<tInput, tInput> tData;
			typedef int tLabel;
			typedef std::pair<tData, tLabel> tSample;
			typedef std::vector<tSample> tSamples;
			tSamples samples;
			{
				for (int i = 0; i < iNumberOfSamples; i++)
				{
					const double dRandom1 = (double)std::rand() / RAND_MAX;
					const double dRandom2 = (double)std::rand() / RAND_MAX;
					const tData data = std::make_pair(dRandom1 > 0.5 ? 1 : 0, dRandom2 > 0.5 ? 1 : 0);
					// XOR data to get label
					const tLabel label = data.first ^ data.second;
					const tSample sample = std::make_pair(data, label);
					samples.push_back(sample);
				}
			}

			// Create new train and test DB
			boost::scoped_ptr<caffe::db::DB> train_db(caffe::db::GetDB(FLAGS_backend));
			std::string dbTrainName = argv[3];
			dbTrainName += "_train";
			train_db->Open(dbTrainName.c_str(), caffe::db::NEW);
			boost::scoped_ptr<caffe::db::Transaction> train_txn(train_db->NewTransaction());

			boost::scoped_ptr<caffe::db::DB> test_db(caffe::db::GetDB(FLAGS_backend));
			std::string dbTestName = argv[3];
			dbTestName += "_test";
			test_db->Open(dbTestName.c_str(), caffe::db::NEW);
			boost::scoped_ptr<caffe::db::Transaction> test_txn(test_db->NewTransaction());

			// divide the train/test data, determine spliting tactic
			const int iSplitRate = std::atoi(argv[8]);;
			int iNumberPutToTrain = 0;
			int iNumberPutToTest = 0;
			enum eNextSample { eNSTrain, eNSTest };
			eNextSample nextSample = eNSTrain; // always start with train samples

											   // convert samples to caffe::Datum
			
			for (tSamples::const_iterator itrSample = samples.begin()
				; itrSample != samples.end()
				; ++itrSample)
			{
				// extract label from sample
				const int iLabel = itrSample->second;

				// convert data to protobuf Datum
				caffe::Datum datum;
				datum.set_channels(2);
				datum.set_height(1);
				datum.set_width(1);
				datum.set_label(iLabel);
				datum.add_float_data(itrSample->first.first);
				datum.add_float_data(itrSample->first.second);

				// write datum to db use the sample number as key for db
				std::string out;
				CHECK(datum.SerializeToString(&out));
				std::stringstream ss;
				ss << iCount;

				// put sample
				if (nextSample == eNSTrain) // always start with train samples
				{
					train_txn->Put(ss.str(), out);
					iNumberPutToTrain++;
					iCountTrain++;
				}
				else if (nextSample == eNSTest)
				{
					test_txn->Put(ss.str(), out);
					iNumberPutToTest++;
					iCountTest++;
				}

				// determine where next sample should go
				if (iSplitRate == 0)
				{
					nextSample = eNSTrain;
				}
				else if (iSplitRate < 0)
				{
					if (iNumberPutToTest == std::abs(iSplitRate))
					{
						nextSample = eNSTrain;
						iNumberPutToTest = 0;
					}
					else
					{
						nextSample = eNSTest;
					}
				}
				else if (iSplitRate > 0)
				{
					if (iNumberPutToTrain == iSplitRate)
					{
						nextSample = eNSTest;
						iNumberPutToTrain = 0;
					}
					else
					{
						nextSample = eNSTrain;
					}
				}

				// every 1000 samples commit to db
				if (iCountTrain % 1000 == 0)
				{
					train_txn->Commit();
					train_txn.reset(train_db->NewTransaction());
				}
				if (iCountTest % 1000 == 0)
				{
					test_txn->Commit();
					test_txn.reset(test_db->NewTransaction());
				}

				iCount++;
			}

			// commit the last unwritten batch
			if (iCountTrain % 1000 != 0)
			{
				train_txn->Commit();
			}
			if (iCountTest % 1000 != 0)
			{
				test_txn->Commit();
			}
		}
		else
		{
			for (int i = 0; i < iLines_1; i++)
			{
				bool status;
				string fn = v_total_1[i][0]; // 文件名

				//	status = ReadFloatData2Datum(v_total_1[i], v_total_2[i], v_total_3[i], bTrainOrTest, 21, 3, 1, &datum, list_buffer);
				//  status = ReadFloatData2Datum_6_Channel_1_2(v_total_1[i], v_total_2[i], v_total_3[i], bTrainOrTest, 6, &datum);
				status = ReadFloatData2Datum_12_Channel_1_1(v_total_1[i], v_total_2[i], v_total_3[i], bTrainOrTest, 12, &datum);

				if (status == false) continue;

				// sequential
				string key_str = caffe::format_int(i, 8) + "_" + fn;

				// Put in db
				string out;
				CHECK(datum.SerializeToString(&out));
				txn->Put(key_str, out);

				if (++count % 1000 == 0)
				{
					// Commit db
					txn->Commit();
					txn.reset(db->NewTransaction());
					LOG(INFO) << "Processed " << count << " files.";
				}
			}// end for
		}		
	}// end else

	// write the last batch
	if (count % 1000 != 0)
	{
		txn->Commit();
		LOG(INFO) << "Processed " << count << " files.";
	}

	for (int i = 0; i < list_buffer.size(); i++)
	{
		delete list_buffer[i];
	}

#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif // USE_OPENCV

	return 0;
}

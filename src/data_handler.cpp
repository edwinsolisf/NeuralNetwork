#include "data_handler.h"

#include "stm/debug.h"

Data::Data(std::string filePath, DATA_TYPE dataType, float* (*readFile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
	:_filePath(filePath), _dataType(dataType), _data(nullptr), _sampleSize(0), _sampleCount(0), ReadFile(readFile)
{
	if(filePath != "")
		_data = ReadFile(filePath.c_str(), _sampleSize, _sampleCount);
}

Data::Data(unsigned int sampleSize, unsigned int sampleCount, DATA_TYPE type, const float* data)
	:_sampleCount(sampleCount), _sampleSize(sampleSize), _dataType(type), _data(new float[sampleSize * sampleCount])
{
	memcpy(_data, data, sizeof(float) * sampleSize * sampleCount);
}

Data::~Data()
{
	delete[] _data;
}

stm::dynamic_vector<float> Data::GetSample(unsigned int id) const
{
	stm_assert(id < _sampleCount);
	return std::move(stm::dynamic_vector<float>(_sampleSize, &_data[id * _sampleSize]));
}

stm::dynamic_matrix<float> Data::GetSampleBatch(const std::vector<unsigned int>& batchIds) const
{
	stm::dynamic_matrix<float> batch(_sampleSize, batchIds.size());
	for (unsigned int i = 0; i < batchIds.size(); ++i)
		batch.SetColumnVector(i, stm::dynamic_vector<float>(_sampleSize, &_data[batchIds[i] * _sampleSize]));
	return batch;
	//return std::move(stm::dynamic_matrix<float>(size, _sampleSize, &_data[_sampleSize * batch * size]));
}

void Data::SetNewData(float* data)
{
	delete _data;
	_data = data;
	//memcpy(_data, data, sizeof(float) * _sampleCount * _sampleSize);
}



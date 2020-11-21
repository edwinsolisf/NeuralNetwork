#include "data_handler.h"
#include "stm/debug.h"

Data::Data(std::string filePath, DATA_TYPE dataType, float* (*readFile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
	:_filePath(filePath), _dataType(dataType), _data(nullptr), _sampleSize(0), _sampleCount(0), ReadFile(readFile)
{
	if(filePath != "")
		_data = ReadFile(filePath.c_str(), _sampleSize, _sampleCount);
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

stm::dynamic_matrix<float> Data::GetSampleBatch(unsigned int batch, unsigned int size) const
{
	stm_assert((batch + 1)* size < _sampleCount);
	return std::move(stm::dynamic_matrix<float>(size, _sampleSize, &_data[_sampleSize * batch * size]));
}



#ifndef data_handler_h
#define data_handler_h

#include <string>
#include <vector>
#include "stm/dynamic_vector.h"
#include "stm/dynamic_matrix.h"

enum class DATA_TYPE
{
	INPUT, OUTPUT
};

class Data
{
public:

	Data(std::string filePath, DATA_TYPE dataType, float* (*readFile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount));
	Data(unsigned int sampleSize, unsigned int sampleCount, DATA_TYPE type, const float* data);
	~Data();
	
	stm::dynamic_vector<float> GetSample(unsigned int id) const;
	stm::dynamic_matrix<float> GetSampleBatch(const std::vector<unsigned int>& batchIds) const;

	unsigned int GetSampleSize() const { return _sampleSize; }
	unsigned int GetSampleCount() const { return _sampleCount; }

	void SetNewData(float* data);
private:
	std::string _filePath;
	DATA_TYPE _dataType;
	float* _data;
	unsigned int _sampleSize;
	unsigned int _sampleCount;

	float* (*ReadFile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount);
};

#endif /* data_handler_h */

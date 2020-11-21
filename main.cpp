#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>

#include "stm/dynamic_matrix.h"
#include "stm/dynamic_vector.h"
#include "stm/utilities.h"

#include "neuralnewtork.h"

#define TEST(x) { long long start = std::chrono::high_resolution_clock::now().time_since_epoch().count();\
for (unsigned int i = 0; i < 10000; ++i) { x; }\
long long end = std::chrono::high_resolution_clock::now().time_since_epoch().count();\
std::cout << (end - start)/10000 << std::endl; }

float* ReadInputFile(const char* filepath, unsigned int& size, unsigned int& count);
float* ReadOutputFile(const char* filepath, unsigned int& size, unsigned int& count);

int main()
{
	NeuralNetwork nn;

	nn.SetUpInputData("assets/data/train-images.idx3-ubyte", ReadInputFile);
	nn.SetUpOutputtData("assets/data/train-labels.idx1-ubyte", ReadOutputFile);
	nn.SetUpTrainingConfiguration(2, 16, 1, 500, 10, 1.0f, 0.0f);
	
	nn.StartTraining();
	Print(nn.ProcessSample(stm::dynamic_vector<float>(784, 1.0f)));
	for (unsigned int i = 0; i < 10; ++i)
	{
		auto sample = nn.TestSample(i);
		Print(sample.first);
		Print(sample.second);
		std::cout << "=================" << std::endl;
	}
	nn.PrintNetworkValues();\

	std::cin.get();
	return 0;
}


float* ReadInputFile(const char* filepath, unsigned int& size, unsigned int& count)
{
	std::ifstream file(filepath, std::ios::binary);

	unsigned int magicNumber;
	unsigned int items;
	unsigned int imageHeight, imageWidth;

	char temp[5];

	file.get(temp[3]);
	file.get(temp[2]);
	file.get(temp[1]);
	file.get(temp[0]);
	memcpy(&magicNumber, temp, 4);

	file.get(temp[3]);
	file.get(temp[2]);
	file.get(temp[1]);
	file.get(temp[0]);
	memcpy(&items, temp, 4);

	file.get(temp[3]);
	file.get(temp[2]);
	file.get(temp[1]);
	file.get(temp[0]);
	memcpy(&imageHeight, temp, 4);

	file.get(temp[3]);
	file.get(temp[2]);
	file.get(temp[1]);
	file.get(temp[0]);
	memcpy(&imageWidth, temp, 4);

	float* data = new float[items * imageHeight * imageWidth];

	for (int i = 0; i < items * imageHeight * imageWidth; i++)
	{
		file.get(temp[0]);
		data[i] = (float)temp[0];
		if (i < imageHeight * imageWidth)
		{
			std::cout << temp[0] << "  ";
			if (i % imageHeight == 0)
				std::cout << std::endl;
		}
	}

	size = imageWidth * imageHeight;
	count = items;
	return data;
}

float* ReadOutputFile(const char* filepath, unsigned int& size, unsigned int& count)
{
	std::ifstream file(filepath, std::ios::binary);

	unsigned int magicNumber;
	unsigned int items;

	char temp[5];

	file.get(temp[3]);
	file.get(temp[2]);
	file.get(temp[1]);
	file.get(temp[0]);
	memcpy(&magicNumber, temp, 4);

	file.get(temp[3]);
	file.get(temp[2]);
	file.get(temp[1]);
	file.get(temp[0]);
	memcpy(&items, temp, 4);

	float* data = new float[items * 10]{ 0 };

	for (int i = 0; i < items; i++)
	{
		file.get(temp[0]);
		data[(i * 10) + temp[0]] = 1.0f;
	}

	size = 10;
	count = items;
	return data;
}
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>

#include "stm/dynamic_matrix.h"
#include "stm/dynamic_vector.h"
#include "stm/utilities.h"

#include "neuralnewtork.h"
#include "avxMath.h"

#define TEST(x) { long long start = std::chrono::high_resolution_clock::now().time_since_epoch().count();\
for (unsigned int i = 0; i < 10000; ++i) { x; }\
long long end = std::chrono::high_resolution_clock::now().time_since_epoch().count();\
std::cout << (end - start)/10000 << std::endl; }

float* ReadInputFile(const char* filepath, unsigned int& size, unsigned int& count);
float* ReadOutputFile(const char* filepath, unsigned int& size, unsigned int& count);

void neural()
{
	NeuralNetwork nn;

	nn.SetUpInputData("assets/data/train-images.idx3-ubyte", ReadInputFile);
	nn.SetUpOutputtData("assets/data/train-labels.idx1-ubyte", ReadOutputFile);
	nn.EnableMultibatch();

	for (unsigned int i = 1; i < 1100; ++i)
	{
		nn.SetUpTrainingConfiguration(2, 16, 5, 20 * i, 24, 0.15f, 0.0f);
		nn.StartTraining();
		std::cout << "\n\nEpoch: " << i << "\n";
		for (unsigned int j = 0; j < 10; ++j)
		{
			auto sample = nn.TestSample(j);
			Print(sample.first);
			Print(sample.second);
			std::cout << "=================\n";
		}
	}

	//Print(nn.ProcessSample(stm::dynamic_vector<float>(784, 1.0f)));
	for (unsigned int i = 0; i < 10; ++i)
	{
		auto sample = nn.TestSample(i);
		Print(sample.first);
		Print(sample.second);
		std::cout << "=================" << std::endl;
	}
}

void Test()
{
	/*float inputs[] =
	{
		1.0f, 1.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
	};*/

	float inputs[] =
	{
		1.0f, 1.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f
	};
	float outputs[] =
	{
		1.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		0.0f, 1.0f,
		0.0f, 1.0f
	};
	/*float outputs[] =
	{
		1.0f,
		0.0f,
		0.0f,
		1.0f,
		1.0f,
		1.0f,
		1.0f,
		0.0f
	};*/

	NeuralNetwork nn;
	nn.SetUpData(8, 3, 2, inputs, outputs);
	nn.EnableMultibatch();
	nn.SetUpTrainingConfiguration(1, 8, 1, 8, 100000, 1.0f, 0.0f);
	auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	nn.StartTraining();
	auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::cout << end - start << " ns\n";
	for (unsigned int i = 0; i < 7; ++i)
	{
		auto sample = nn.TestSample(i);
		Print(sample.first);
		Print(sample.second);
		std::cout << "=================" << std::endl;
	}
	nn.PrintNetworkValues();
}

int main()
{
	//Test();
	//stm::dynamic_vector<float> dv(100, 5.0f);
	//stm::dynamic_matrix<float> dm(16, 100, 1.1f);
	//stm::dynamic_vector<float> dv2(16, 1.0f);

	//dv2 = stm::multiply(dm, dv);

	neural();
	float arr[] =
	{
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	stm::mat4f m1(arr);
	stm::mat4f m2(arr);
	
	//stm::dynamic_matrix<float> dm(4, 4, 2.0f);
	//stm::dynamic_vector<float> dv(4, 1.0f);

	//stm::dynamic_matrix<float> dm(16, 784, 1.1f);
	//stm::dynamic_matrix<float> dm2(784, 20, 5.1f);
	//stm::dynamic_vector<float> dv(100, 5.0f);
	//stm::dynamic_vector<float> dv2(100, 1.0f);

	//Print(stm::multiply(dm, dv));
	//Print(avx::multiply(dm, dv));
	//Print(avx::multiply256(dm, dv));
	//TEST(stm::multiply(dm, dv));
	//TEST(avx::multiply(dm, dv));
	//TEST(avx::multiply256(dm, dv));
	//Print(dv + dv2);
	//Print(avx::add(dv, dv2));
	//Print(stm::multiply(dm, dm2));
	//std::cout << "==============\n";
	//Print(avx::multiply(dm, dm2));

	//TEST(stm::multiply(dm, dm2));
	//TEST(avx::multiply(dm, dm2));
	//TEST(dv * dv2);
	//TEST(avx::dot(dv, dv2));


	Print(stm::multiply(m1, m2));
	Print(avx::multiply(m1, m2));
	TEST(stm::multiply(m1, m2));
	TEST(avx::multiply(m1, m2));

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
		data[i] = (float)(int)(unsigned char)temp[0];
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
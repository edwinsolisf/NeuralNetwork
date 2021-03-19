#include "my_network.h"

#include <fstream>
#include <chrono>

#include "stm/dynamic_matrix.h"
#include "stm/dynamic_vector.h"
#include "stm/utilities.h"

#include "static_neural_network.h"
#include "neuralnewtork.h"
//#include "stm/avx_math.h"


static float* ReadInputFile(const char* filepath, unsigned int& size, unsigned int& count);
static float* ReadOutputFile(const char* filepath, unsigned int& size, unsigned int& count);

void neural()
{
	NeuralNetwork nn;
	
	nn.SetUpInputData("assets/data/train-images.idx3-ubyte", ReadInputFile);
	nn.SetUpOutputData("assets/data/train-labels.idx1-ubyte", ReadOutputFile);
	//nn.EnableMultibatch();

	/*for (unsigned int i = 1; i < 1100; ++i)
	{
		nn.SetUpTrainingConfiguration(2, 16, 8, 50 * i, 8, 0.75f, 0.9f);
		nn.StartTraining();
		std::cout << "\n\nEpoch: " << i << "\n";
		for (unsigned int j = 0; j < 20; ++j)
		{
			auto sample = nn.TestSample(j);
			Print(sample.first);
			Print(sample.second);
			std::cout << "=================\n";
		}
	}*/

	for (unsigned int i = 1; i < 500; ++i)
	{
		//For 1 batch, learning rate = 0.5f
		//For 4 batch, learning rate = 0.75f
		//For 8 batch, learning rate = 0.95f
		nn.SetUpTrainingConfiguration(1, 30, 10, 50000, 1, 1.5f, 0.9f);
		
		auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		nn.StartTraining();
		auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::cout << "\n\nEpoch: " << i << "\n" << "Time: " << (end - start)/1000000.0f << "\n";
		for (unsigned int j = 0; j < 20; ++j)
		{
			auto sample = nn.TestSample(j);
			Print(sample.first);
			Print(sample.second);
			std::cout << "=================\n";
		}

	}

	for (unsigned int i = 0; i < 10; ++i)
	{
		auto sample = nn.TestSample(i);
		Print(sample.first);
		Print(sample.second);
		std::cout << "=================" << std::endl;
	}
}

void neuralstatic()
{
	using Network = StaticNeuralNetwork<784, 10, 1, 64>;
	Network& nn = *new Network;

	nn.SetUpInputData("assets/data/train-images.idx3-ubyte", ReadInputFile);
	nn.SetUpOutputData("assets/data/train-labels.idx1-ubyte", ReadOutputFile);
	nn.EnableMultibatch();
	
	auto GetGreatIndex = [](const Network::OutData_t& vec)
	{
		int highest = 0;
		for (unsigned int i = 1; i < vec.GetSize(); ++i)
			if (vec[highest] < vec[i]) highest = i;
		return highest;
	};
	/*for (unsigned int i = 0; i < 20; ++i)
	{
		auto pair = nn.GetSampleData(i+2000);
		auto& input = pair.first;
		auto& output = pair.second;

		stm::Print(stm::matrix<float, 28, 28>(input.GetData()));
		stm::Print(output);
	}*/

	/*for (unsigned int i = 1; i < 1100; ++i)
	{
		nn.SetUpTrainingConfiguration(8, 50 * i, 8, 0.75f, 0.9f);
		nn.StartTraining();
		std::cout << "\n\nEpoch: " << i << "\n";
		for (unsigned int j = 0; j < 10; ++j)
		{
			auto sample = nn.TestSample(j);
			Print(sample.first);
			Print(sample.second);
			std::cout << "=================\n";
		}
	}*/

	unsigned int offset = 50000;
	nn.SetUpTrainingConfiguration(10, 50000 , 1, 3.0f, 1.0f);
	for (unsigned int i = 1; i < 500; ++i)
	{
		auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		nn.StartTraining();
		auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::cout << "\n\nEpoch: " << i << "\n" << "Time: " << (end - start) / 1000000.0f << "\n";

		for (unsigned int j = 0; j < 10; ++j)
		{
			auto sample = nn.TestSample(j);
			Print(sample.first);
			Print(sample.second);
			std::cout << "=================\n";
		}
		unsigned int counter = 0;
		unsigned int counter2 = 0;
		for (unsigned int i = 0; i < 10000; ++i)
		{
			if (nn.GetCost(i + offset).ApplyToVector(sqrt).Magnitude() < 0.316f) counter++;
			auto s = nn.TestSample(i + offset);
			if (GetGreatIndex(s.first) == GetGreatIndex(s.second))
				counter2++;
			
		}
		std::cout << "\n\t" << counter << " / 10000 samples\n\n";
		std::cout << "\n\t" << counter2 << " / 10000 samples\n\n";

	}


	delete& nn;
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
	
	nn.SetUpTrainingConfiguration(2, 10, 8, 8, 100000, 0.1f, 0.0f);
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

void TestStatic()
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

	StaticNeuralNetwork<3, 2, 1, 8> nn;
	nn.SetUpData(8, inputs, outputs);
	nn.EnableMultibatch();

	nn.SetUpTrainingConfiguration(1, 8, 100000, 1.0f, 0.9f);
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
		data[i] = ((float)(int)(unsigned char)temp[0]) / 255.0f;
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

void reverseneural()
{
	using Network = StaticNeuralNetwork<10, 784, 1, 100>;
	Network& nn = *new Network;

	nn.SetUpOutputData("assets/data/train-images.idx3-ubyte", ReadInputFile);
	nn.SetUpInputData("assets/data/train-labels.idx1-ubyte", ReadOutputFile);
	nn.EnableMultibatch();

	auto GetGreatIndex = [](const Network::OutData_t& vec)
	{
		int highest = 0;
		for (unsigned int i = 1; i < vec.GetSize(); ++i)
			if (vec[highest] < vec[i]) highest = i;
		return highest;
	};

	/*for (unsigned int i = 1; i < 1100; ++i)
	{
		nn.SetUpTrainingConfiguration(8, 50 * i, 8, 0.75f, 0.9f);
		nn.StartTraining();
		std::cout << "\n\nEpoch: " << i << "\n";
		for (unsigned int j = 0; j < 10; ++j)
		{
			auto sample = nn.TestSample(j);
			Print(sample.first);
			Print(sample.second);
			std::cout << "=================\n";
		}
	}*/

	unsigned int offset = 50000;
	nn.SetUpTrainingConfiguration(1, 60000, 1, 0.5f, 1.05f);
	for (unsigned int i = 1; i < 500; ++i)
	{

		nn.StartTraining();
		std::cout << "\n\nEpoch: " << i << "\n";
		for (unsigned int j = 0; j < 10; ++j)
		{
			auto sample = nn.TestSample(j);
			Print(stm::matrix<int, 28, 28>((sample.first * 256.0f).Cast<int>().GetData()));
			Print(stm::matrix<int, 28, 28>((sample.second * 256.0f).Cast<int>().GetData()));
			std::cout << "=================\n";
		}
		/*unsigned int counter = 0;
		unsigned int counter2 = 0;
		for (unsigned int i = 0; i < 10000; ++i)
		{
			if (nn.GetCost(i + offset).ApplyToVector(sqrt).Magnitude() < 0.316f) counter++;
			auto s = nn.TestSample(i + offset);
			if (GetGreatIndex(s.first) == GetGreatIndex(s.second))
				counter2++;

		}
		std::cout << "\n\t" << counter << " / 10000 samples\n\n";
		std::cout << "\n\t" << counter2 << " / 10000 samples\n\n";*/

	}


	delete& nn;
}
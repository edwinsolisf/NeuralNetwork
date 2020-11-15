#include <iostream>
#include <thread>
#include <chrono>

#include "stm/dynamic_matrix.h"
#include "stm/utilities.h"

int main()
{
	stm::dynamic_matrix<float> dmat1(100, 100, 2.0f);
	stm::dynamic_matrix<float> dmat2(100, 100, 1400.1f);
	stm::dynamic_matrix<float> result2(100, 100);

	long long avg = 0;
	for (unsigned int j = 0; j < 100; j++)
	{
		auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		for (unsigned int i = 0; i < 10000; ++i)
		{
			result2 = dmat1 + dmat2;
		}
		auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::cout << (end - start)/10000 << " ns \n";
		avg += (end - start) / 10000;
	}
	std::cout << avg / 100 << " ns" << std::endl;
	std::cout << "Hello World!" << std::endl;

	std::cin.get();

	return 0;
}
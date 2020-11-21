#ifndef stm_distribution_h
#define stm_distribution_h

#include <random>
#include <chrono>

namespace stm
{
	float normaldistr_randomf(float mean = 0.0f, float standard_deviation = 1.0f)
	{
		srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<float> distribution(mean, standard_deviation);
		std::default_random_engine generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		return distribution(generator);
	}

	double normaldistr_random(double mean = 0.0, double standard_deviation = 1.0)
	{
		srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		std::normal_distribution<double> distribution(mean, standard_deviation);
		std::default_random_engine generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		return distribution(generator);
	}
}

#endif /* stm_distribution_h */

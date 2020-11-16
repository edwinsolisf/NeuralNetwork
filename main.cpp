#include <iostream>
#include <thread>
#include <chrono>

#include "stm/dynamic_matrix.h"
#include "stm/dynamic_vector.h"
#include "stm/utilities.h"

#define TEST(x) { long long start = std::chrono::high_resolution_clock::now().time_since_epoch().count();\
for (unsigned int i = 0; i < 10000; ++i) { x; }\
long long end = std::chrono::high_resolution_clock::now().time_since_epoch().count();\
std::cout << (end - start)/10000 << std::endl; }


int main()
{
	/*
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
		std::cout << avg / 100 << " ns" << std::endl;
	}*/
	float marr[16] = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f
	};

	float varr[4] = {
		-1.0f, 2.0f, 3.0f, -4.0f
	};

	stm::mat4f mat(marr);
	stm::vec4f vec(varr);
	stm::Print(mat);
	stm::Print(vec);

	auto out = mat.Multiply(vec);
	stm::Print(out);
	std::cout << mat.Determinant() << std::endl;

	stm::mat_f dmat(4, 4, marr);
	stm::vec_f dvec(4, varr);
	stm::Print(dmat);
	stm::Print(dvec);

	auto dout = dmat.Multiply(dvec);
	stm::Print(dout);
	std::cout << dmat.Determinant() << std::endl;

	std::cout << "Hello World!" << std::endl;
	TEST(mat.Determinant());
	TEST(dmat.Determinant());
	TEST(stm::multiply(mat, mat));
	TEST(stm::multiply(dmat, dmat));
	TEST(stm::multiply(mat, dmat));
	TEST(stm::multiply(dmat, mat));
	std::cout << "========================\n";
	TEST(dmat = dmat + mat);
	stm::mat2f mat2(1.0f);
	stm::mat_f dmat2(2, 2, 1.0f);

	mat2 = dmat2;
	stm::vector<int, 5> v;
	stm::vec_i dv(5);

	v = dv;

	TEST(mat += dmat);
	TEST(dmat += mat);
	TEST(dmat + mat);
	TEST(mat + dmat);

	std::cin.get();

	return 0;
}
#include <iostream>

#include "my_network.h"
#include "stm/aligned_matrix.h"
#include "stm/utilities.h"
#include "avxMath.h"

int main()
{
	//neural();
	//Test();
	//neuralstatic();
	//TestStatic();
	//reverseneural();

	const unsigned int size = 100;
	stm::matrix<float, size, size> mat1(13.2f), mat2(-0.29f);
	stm::dynamic_matrix<float> dm1(size, size, 13.2f), dm2(size, size, -0.29f);
	stm::aligned_matrix<float> m1(size, size, 13.2f), m2(size, size, -0.29f);

	TEST(dm1 + dm2 + dm2);
	//TEST(dm1 + dm2 + dm2);
	/*stm::Print(mat1.Multiply(mat2));
	stm::Print(mat1.mult(mat2));
	//stm::Print(avx::multiply(mat1, mat2));
	stm::Print(dm1.Multiply(dm2));
	stm::Print(dm1.mult(dm2));
	//stm::Print(avx::multiply(dm1, dm2));
	stm::Print(m1.Multiply(m2).ToDynamic());
	stm::Print(avx::multiply(m1, m2).ToDynamic());
	stm::Print(m1.mult(m2).ToDynamic());
	stm::Print(avx::multiply256(m1, m2).ToDynamic());
	*/
	/*TEST(mat1.Multiply(mat2));
	TEST(mat1.mult(mat2));
	//TEST(avx::multiply(mat1, mat2));
	TEST(dm1.Multiply(dm2));
	TEST(dm1.mult(dm2));
	//TEST(avx::multiply(dm1, dm2));
	TEST(m1.Multiply(m2));
	TEST(avx::multiply(m1, m2));
	TEST(m1.mult(m2));
	TEST(avx::multiply256(m1, m2));*/

	std::cin.get();
	return 0;
}



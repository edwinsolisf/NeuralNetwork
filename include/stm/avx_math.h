#ifndef stm_avx_math_h
#define stm_avx_math_h

#include "stm/matrix.h"
#include "stm/vector.h"
#include "stm/dynamic_matrix.h"
#include "stm/dynamic_vector.h"
#include "stm/aligned_matrix.h"

namespace stm
{
	namespace avx
	{
		static stm::mat4f dot(const stm::mat4f& mat1, const stm::mat4f& mat2);

		static stm::mat4f multiply(const stm::mat4f& mat1, const stm::mat4f& mat2);

		static stm::dynamic_vector<float> multiply(const stm::dynamic_matrix<float>& mat, const stm::dynamic_vector<float>& vec);

		static stm::dynamic_matrix<float> multiply(const stm::dynamic_matrix<float>& mat1, const stm::dynamic_matrix<float>& mat2);

		static stm::dynamic_vector<float> multiply256(const stm::dynamic_matrix<float>& mat, const stm::dynamic_vector<float>& vec);

		static stm::dynamic_vector<float> add(const stm::dynamic_vector<float>& vec1, const stm::dynamic_vector<float>& vec2);

		static stm::dynamic_vector<float> dot(const stm::dynamic_vector<float>& vec1, const stm::dynamic_vector<float>& vec2);

		template<unsigned int _ROWS, unsigned int _COLUMNS>
		stm::vector<float, _ROWS> multiply256(const stm::matrix<float, _ROWS, _COLUMNS>& mat, const stm::vector<float, _COLUMNS>& vec);

		static stm::aligned_matrix<float> add(const stm::aligned_matrix<float>& m1, const stm::aligned_matrix<float>& m2);

		static stm::aligned_matrix<float> multiply(const stm::aligned_matrix<float>& m1, const stm::aligned_matrix<float>& m2);

		static stm::aligned_matrix<float> multiply256(const stm::aligned_matrix<float>& m1, const stm::aligned_matrix<float>& m2);
	}
}

#include "avx_math.ipp"

#endif /* stm_avx_math_h */
#include "stm/matrix.h"
#include "stm/vector.h"
#include "stm/dynamic_matrix.h"
#include "stm/dynamic_vector.h"

namespace avx
{
	stm::mat4f dot(const stm::mat4f& mat1, const stm::mat4f& mat2);
	stm::mat4f multiply(const stm::mat4f& mat1, const stm::mat4f& mat2);

	stm::dynamic_vector<float> multiply(const stm::dynamic_matrix<float>& mat, const stm::dynamic_vector<float>& vec);
	stm::dynamic_vector<float> multiply256(const stm::dynamic_matrix<float>& mat, const stm::dynamic_vector<float>& vec);
	stm::dynamic_vector<float> add(const stm::dynamic_vector<float>& vec1, const stm::dynamic_vector<float>& vec2);
	stm::dynamic_vector<float> dot(const stm::dynamic_vector<float>& vec1, const stm::dynamic_vector<float>& vec2);
	stm::dynamic_matrix<float> multiply(const stm::dynamic_matrix<float>& mat1, const stm::dynamic_matrix<float>& mat2);


}

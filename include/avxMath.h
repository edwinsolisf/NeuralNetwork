#ifndef avx_math_h
#define avx_math_h

#include "stm/matrix.h"
#include "stm/vector.h"
#include "stm/dynamic_matrix.h"
#include "stm/dynamic_vector.h"
#include <immintrin.h>
#include <memory>
#include <chrono>

namespace avx
{
	static void set(const stm::mat4f& mat, __m256* avxRegister)
	{
		for (unsigned int i = 0; i < 2; ++i)
			avxRegister[i] = _mm256_set_ps(mat[i * 2][0], mat[i * 2][1], mat[i * 2][2], mat[i * 2][3], mat[i * 2][4], mat[i * 2][5], mat[i * 2][6], mat[i * 2][7]);
	}

	static stm::mat4f dot(const stm::mat4f& mat1, const stm::mat4f& mat2)
	{
		//__m128 values = _mm_set_ps(mat1[i][0], mat1[i][1], mat1[i][2], mat1[i][3]);
		stm::mat4f temp;

		__m256 values1[2], values2[2];
		//set(mat1, values1);
		//set(mat2, values2);
		values1[0] = _mm256_set_ps(mat1[0 * 2][0], mat1[0 * 2][1], mat1[0 * 2][2], mat1[0 * 2][3], mat1[0 * 2][4], mat1[0 * 2][5], mat1[0 * 2][6], mat1[0 * 2][7]);
		values1[1] = _mm256_set_ps(mat1[1 * 2][0], mat1[1 * 2][1], mat1[1 * 2][2], mat1[1 * 2][3], mat1[1 * 2][4], mat1[1 * 2][5], mat1[1 * 2][6], mat1[1 * 2][7]);
		values2[0] = _mm256_set_ps(mat2[0 * 2][0], mat2[0 * 2][1], mat2[0 * 2][2], mat2[0 * 2][3], mat2[0 * 2][4], mat2[0 * 2][5], mat2[0 * 2][6], mat2[0 * 2][7]);
		values2[1] = _mm256_set_ps(mat2[1 * 2][0], mat2[1 * 2][1], mat2[1 * 2][2], mat2[1 * 2][3], mat2[1 * 2][4], mat2[1 * 2][5], mat2[1 * 2][6], mat2[1 * 2][7]);

		__m256 ans = _mm256_mul_ps(values1[0], values2[0]);
		memcpy(temp[0], (float*)&ans, 8 * sizeof(float));
		ans = _mm256_mul_ps(values1[1], values2[1]);
		memcpy(temp[2], (float*)&ans, 8 * sizeof(float));
		return temp;
	}

	static void setColumnRegisters(const stm::mat4f& mat, __m128* avxRegister)
	{
		for (unsigned int i = 0; i < 4; ++i)
			avxRegister[i] = _mm_set_ps(mat[0][i], mat[1][i], mat[2][i], mat[3][i]);
	}

	static void setRowRegisters(const stm::mat4f& mat, __m128* avxRegister)
	{
		for (unsigned int i = 0; i < 4; ++i)
			avxRegister[i] = _mm_set_ps(mat[i][0], mat[i][1], mat[i][2], mat[i][3]);
	}

	static stm::mat4f multiply(const stm::mat4f& mat1, const stm::mat4f& mat2)
	{
		__m128 val1[4], val2[4];
		setRowRegisters(mat1, val1);
		setColumnRegisters(mat2, val2);

		__m128 vals[16];
		stm::mat4f temp;

		for (unsigned int i = 0; i < 4; ++i)
		{
			for (unsigned int j = 0; j < 4; ++j)
				vals[i * 4 + j] = _mm_mul_ps(val1[i], val2[j]);

			__m128 v01 = _mm_hadd_ps(vals[i * 4], vals[i * 4 + 1]);
			__m128 v23 = _mm_hadd_ps(vals[i * 4 + 2], vals[i * 4 + 3]);

			//v01 = _mm_permute_ps(v01, 0b11100100);
			//v23 = _mm_permute_ps(v23, 0b11100100);

			__m128 res = _mm_hadd_ps(v01, v23);
			//res = _mm_permute_ps(res, 0b11100100);
			memcpy(temp[i], (float*)&res, 4 * sizeof(float));
		}

		return temp;
	}

	static __m128 addIndvRegisters(__m128 v0, __m128 v1, __m128 v2, __m128 v3)
	{
		__m128 v01 = _mm_hadd_ps(v0, v1);
		__m128 v23 = _mm_hadd_ps(v2, v3);

		//v01 = _mm_permute_ps(v01, 0b11100100);
		//v23 = _mm_permute_ps(v23, 0b11100100);

		__m128 res = _mm_hadd_ps(v01, v23);
		//res = _mm_permute_ps(res, 0b11100100);
		return res;
	}

	static float addIndvRegister(__m128 v)
	{
		__m128 t1 = _mm_hadd_ps(v, v);
		__m128 t2 = _mm_hadd_ps(t1, t1);
		return _mm_cvtss_f32(t2);
	}

	static __m128 setRegisterValues(int count, const float data[4])
	{
		switch (count)
		{
		case 1:
			return _mm_set_ps(data[0], 0.0f, 0.0f, 0.0f);
		case 2:
			return _mm_set_ps(data[0], data[1], 0.0f, 0.0f);
		case 3:
			return _mm_set_ps(data[0], data[1], data[2], 0.0f);
		case 4:
			return _mm_set_ps(data[0], data[1], data[2], data[3]);
		default:
			return _mm_set_ps1(0.0f);
		}
	}



	static __m256 load_data(int count, const float data[8])
	{
		switch (count)
		{
		case 1:
			return _mm256_set_ps(data[0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 2:
			return _mm256_set_ps(data[0], data[1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 3:
			return _mm256_set_ps(data[0], data[1], data[2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		case 4:
			return _mm256_set_ps(data[0], data[1], data[2], data[3], 0.0f, 0.0f, 0.0f, 0.0f);
		case 5:
			return _mm256_set_ps(data[0], data[1], data[2], data[3], data[4], 0.0f, 0.0f, 0.0f);
		case 6:
			return _mm256_set_ps(data[0], data[1], data[2], data[3], data[4], data[5], 0.0f, 0.0f);
		case 7:
			return _mm256_set_ps(data[0], data[1], data[2], data[3], data[4], data[5], data[6], 0.0f);
		case 8:
			return _mm256_loadu_ps(data);
		default:
			return _mm256_set1_ps(0.0f);
		}
	}

	static inline __m256 load_data(const float data[8])
	{
		return _mm256_load_ps(data);
	}

	static __m256 addRegisters(__m256 v0, __m256 v1, __m256 v2, __m256 v3, __m256 v4, __m256 v5, __m256 v6, __m256 v7)
	{
		__m256 v01 = _mm256_hadd_ps(v0, v1);
		__m256 v23 = _mm256_hadd_ps(v2, v3);
		__m256 v45 = _mm256_hadd_ps(v4, v5);
		__m256 v67 = _mm256_hadd_ps(v6, v7);

		__m256 v0123 = _mm256_hadd_ps(v01, v23);
		__m256 v4567 = _mm256_hadd_ps(v45, v67);

		return _mm256_hadd_ps(v0123, v4567);
	}

	static float addRegister(__m256 v)
	{
		__m128 t1 = _mm256_castps256_ps128(v);
		__m128 t2 = _mm256_extractf128_ps(v, 1);
		return addIndvRegister(_mm_add_ps(t1, t2));
	}

	static __m128 hadd128(const __m128* v)
	{
		__m128 v01 = _mm_hadd_ps(v[0], v[1]);
		__m128 v23 = _mm_hadd_ps(v[2], v[3]);

		return _mm_hadd_ps(v01, v23);
	}

	static __m256 hadd256(const __m256* v)
	{
		__m256i control = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
		__m256 v01 = _mm256_hadd_ps(v[0], v[1]);
		v01 = _mm256_permutevar8x32_ps(v01, control);

		__m256 v23 = _mm256_hadd_ps(v[2], v[3]);
		v23 = _mm256_permutevar8x32_ps(v23, control);

		__m256 v45 = _mm256_hadd_ps(v[4], v[5]);
		v45 = _mm256_permutevar8x32_ps(v45, control);

		__m256 v67 = _mm256_hadd_ps(v[6], v[7]);
		v67 = _mm256_permutevar8x32_ps(v67, control);

		__m256 v0123 = _mm256_hadd_ps(v01, v23);
		v01 = _mm256_permutevar8x32_ps(v0123, control);

		__m256 v4567 = _mm256_hadd_ps(v45, v67);

		v4567 = _mm256_permutevar8x32_ps(v4567, control);

		__m256 res = _mm256_hadd_ps(v0123, v4567);
		return _mm256_permutevar8x32_ps(res, control);
	}

	static inline __m128 load_data128(const float data[4])
	{
		return _mm_loadu_ps(data);
	}

	static stm::dynamic_vector<float> multiply(const stm::dynamic_matrix<float>& mat, const stm::dynamic_vector<float>& vec)
	{
		stm_assert(mat.GetColumnSize() == vec.GetSize());
		//auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		unsigned int rows = mat.GetRowSize();
		unsigned int columns = mat.GetColumnSize();
		unsigned int num = columns % 16;
		unsigned int k = (columns) / 16;
		unsigned int t = (k * 16);
		stm::dynamic_vector<float> out(rows);
		__m128 sum;
		__m128 values[4];
		//auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		//std::cout << end - start << " ns\n";

		//start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		for (unsigned int i = 0; i < rows; ++i)
		{
			int z = num;
			sum = _mm_setzero_ps();
			for (unsigned int j = 0; j != k; ++j)
			{
				values[0] = _mm_mul_ps(load_data128(&mat[i][((j * 4) + 0) * 4]), load_data128(&vec[((j * 4) + 0) * 4]));
				values[1] = _mm_mul_ps(load_data128(&mat[i][((j * 4) + 1) * 4]), load_data128(&vec[((j * 4) + 1) * 4]));
				values[2] = _mm_mul_ps(load_data128(&mat[i][((j * 4) + 2) * 4]), load_data128(&vec[((j * 4) + 2) * 4]));
				values[3] = _mm_mul_ps(load_data128(&mat[i][((j * 4) + 3) * 4]), load_data128(&vec[((j * 4) + 3) * 4]));
				sum = _mm_add_ps(sum, hadd128(values));
			}

			for (unsigned int n = 0; n < 4; ++n)
			{
				if ((z - 4) >= 0)
					values[n] = _mm_mul_ps(load_data128(&mat[i][t + (n * 4)]), load_data128(&vec[t + (n * 4)]));
				else
#ifndef DEBUG
					values[n] = _mm_mul_ps(setRegisterValues(z, &mat[i][t + (n * 4)]), setRegisterValues(z, &vec[t + (n * 4)]));
#else
					values[n] = _mm_mul_ps(setRegisterValues(z, &mat[i][t + (n * 4)]), setRegisterValues(z, &vec[(t + (n * 4) >= columns) ? 0 : (t + (n * 4))]));
#endif
				z -= 4;
			}

			sum = _mm_add_ps(sum, hadd128(values));
			/*values[0] = _mm_mul_ps(setRegisterValues(num > 4 ? 4 : num, &mat[i][(k * 4)]), setRegisterValues(4, &vec[(k * 4)]));
			values[1] = _mm_mul_ps(setRegisterValues((num - 4) > 4 ? 4 : (num - 4), &mat[i][(k * 4) + 1]), setRegisterValues(4, &vec[(k * 4) + 1]));
			values[2] = _mm_mul_ps(setRegisterValues((num - 8) > 4 ? 4 : (num - 8), &mat[i][(k * 4) + 2]), setRegisterValues(4, &vec[(k * 4) + 2]));
			values[3] = _mm_mul_ps(setRegisterValues((num - 12) > 4 ? 4 : (num - 12), &mat[i][(k * 4) + 3]), setRegisterValues(4, &vec[(k * 4) + 3]));
			sum = _mm_add_ps(sum, addIndvRegisters(values[0], values[1], values[2], values[3]));*/

			out[i] = addIndvRegister(sum);
		}
		//end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		//std::cout << end - start << " ns\n";

		return out;
	}

	static stm::dynamic_matrix<float> multiply(const stm::dynamic_matrix<float>& mat1, const stm::dynamic_matrix<float>& mat2)
	{
		stm::dynamic_matrix<float> out(mat1.GetRowSize(), mat2.GetColumnSize());
		for (unsigned int y = 0; y < mat2.GetColumnSize(); ++y)
		{
			for (unsigned int x = 0; x < mat1.GetRowSize(); ++x)
			{
				unsigned int rows = mat1.GetRowSize();
				unsigned int columns = mat1.GetColumnSize();
				unsigned int num = columns % 16;
				unsigned int k = (columns) / 16;
				unsigned int t = (k * 16) - 1;
				__m128 sum;
				__m128 values[4];

				for (unsigned int i = 0; i < rows; ++i)
				{
					int z = num;
					sum = _mm_setzero_ps();
					for (unsigned int j = 0; j != k; ++j)
					{
						values[0] = _mm_mul_ps(load_data128(&mat1[i][((j * 4) + 0) * 4]), load_data128(&mat2[y][((j * 4) + 0) * 4]));
						values[1] = _mm_mul_ps(load_data128(&mat1[i][((j * 4) + 1) * 4]), load_data128(&mat2[y][((j * 4) + 1) * 4]));
						values[2] = _mm_mul_ps(load_data128(&mat1[i][((j * 4) + 2) * 4]), load_data128(&mat2[y][((j * 4) + 2) * 4]));
						values[3] = _mm_mul_ps(load_data128(&mat1[i][((j * 4) + 3) * 4]), load_data128(&mat2[y][((j * 4) + 3) * 4]));
						sum = _mm_add_ps(sum, hadd128(values));
					}

					for (unsigned int n = 0; n < 4; ++n)
					{
						if ((z - 4) >= 0)
							values[n] = _mm_mul_ps(load_data128(&mat1[i][t + (n * 4)]), load_data128(&mat2[y][t + (n * 4)]));
						else
#ifndef DEBUG
							values[n] = _mm_mul_ps(setRegisterValues(z, &mat1[i][t + (n * 4)]), setRegisterValues(z, &mat2[y][t + (n * 4)]));
#else
							values[n] = _mm_mul_ps(setRegisterValues(z, &mat1[i][t + (n * 4)]), setRegisterValues(z, &mat2[y][(t + (n * 4) >= columns) ? 0 : (t + (n * 4))]));
#endif
						z -= 4;
					}

					sum = _mm_add_ps(sum, hadd128(values));

					out[x][y] = addIndvRegister(sum);
				}
			}
		}
		return out;
	}

	static stm::dynamic_vector<float> multiply256(const stm::dynamic_matrix<float>& mat, const stm::dynamic_vector<float>& vec)
	{
		stm_assert(mat.GetColumnSize() == vec.GetSize());

		unsigned int rows = mat.GetRowSize();
		unsigned int columns = mat.GetColumnSize();
		unsigned int num = columns % 64;
		unsigned int k = columns / 64;
		unsigned int t = (k * 64);
		stm::dynamic_vector<float> out(rows);
		__m256 sum;
		__m256 values[8];

		for (unsigned int i = 0; i < rows; ++i)
		{
			int z = num;
			sum = _mm256_setzero_ps();
			for (unsigned int j = 0; j != k; ++j)
			{
				values[0] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 0) * 8]), load_data(&vec[((j * 8) + 0) * 8]));
				values[1] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 1) * 8]), load_data(&vec[((j * 8) + 1) * 8]));
				values[2] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 2) * 8]), load_data(&vec[((j * 8) + 2) * 8]));
				values[3] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 3) * 8]), load_data(&vec[((j * 8) + 3) * 8]));
				values[4] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 4) * 8]), load_data(&vec[((j * 8) + 4) * 8]));
				values[5] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 5) * 8]), load_data(&vec[((j * 8) + 5) * 8]));
				values[6] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 6) * 8]), load_data(&vec[((j * 8) + 6) * 8]));
				values[7] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 7) * 8]), load_data(&vec[((j * 8) + 7) * 8]));

				sum = _mm256_add_ps(sum, hadd256(values));
			}

			for (unsigned int n = 0; n < 8; ++n)
			{
				if ((z - 8) >= 0)
					values[n] = _mm256_mul_ps(load_data(&mat[i][t + (n * 8)]), load_data(&vec[t + (n * 8)]));
				else
#ifndef DEBUG
					values[n] = _mm256_mul_ps(load_data(z, &mat[i][t + (n * 8)]), load_data(z, &vec[t + (n * 8)]));
#else
					values[n] = _mm256_mul_ps(load_data(z, &mat[i][t + (n * 8)]), load_data(z, &vec[(t + (n * 8) >= columns) ? 0 : (t + (n * 8))]));
#endif
				z -= 8;
			}
			sum = _mm256_add_ps(sum, hadd256(values));

			out[i] = addRegister(sum);
		}

		return out;
	}

	static stm::dynamic_vector<float> add(const stm::dynamic_vector<float>& vec1, const stm::dynamic_vector<float>& vec2)
	{
		unsigned int n = vec1.GetSize();
		stm::dynamic_vector<float> out(n);
		unsigned int k = ((n + 8) / 8) - 1;
		unsigned int z = n - k * 8;
		unsigned int t = k * 8;
		for (unsigned int i = 0; i != k; ++i)
		{
			unsigned int ind = i * 8;
			__m256 val1 = _mm256_loadu_ps(&vec1[ind]);
			__m256 val2 = _mm256_loadu_ps(&vec2[ind]);
			__m256 sum = _mm256_add_ps(val1, val2);

			memcpy(&out[ind], (float*)&sum, sizeof(float) * 8);
		}
		if (z)
		{
			__m256 val1 = load_data(z, &vec1[t]);
			__m256 val2 = load_data(z, &vec2[t]);
			__m256 sum = _mm256_add_ps(val1, val2);
			memcpy(&out[t], (float*)&sum, sizeof(float) * z);
		}
		return out;
	}

	static stm::dynamic_vector<float> dot(const stm::dynamic_vector<float>& vec1, const stm::dynamic_vector<float>& vec2)
	{
		unsigned int n = vec1.GetSize();
		stm::dynamic_vector<float> out(n);
		unsigned int k = ((n + 8) / 8) - 1;
		unsigned int z = n - k * 8;
		unsigned int t = k * 8;
		for (unsigned int i = 0; i != k; ++i)
		{
			unsigned int ind = i * 8;
			__m256 val1 = _mm256_loadu_ps(&vec1[ind]);
			__m256 val2 = _mm256_loadu_ps(&vec2[ind]);
			__m256 sum = _mm256_mul_ps(val1, val2);

			memcpy(&out[ind], (float*)&sum, sizeof(float) * 8);
		}
		if (z)
		{
			__m256 val1 = load_data(z, &vec1[t]);
			__m256 val2 = load_data(z, &vec2[t]);
			__m256 sum = _mm256_mul_ps(val1, val2);
			memcpy(&out[t], (float*)&sum, sizeof(float) * z);
		}
		return out;
	}

	/*stm::dynamic_matrix<float> mult(const stm::dynamic_matrix<float>& mat1, const stm::dynamic_matrix<float>& mat2)
	{
		stm::dynamic_matrix<float> out(mat1.GetRowSize(), mat2.GetColumnSize());
		stm::dynamic_matrix<float> m1 = mat1.Transpose();
		for (unsigned int i = 0; i < mat1.GetColumnSize(); ++i)
		{
			__m256 sum = _mm256_setzero_ps();

			__m256 matrixRow;
			__m256 vectorRow;
			sum = _mm256_fmadd_ps(matrixRow, vectorRow, sum);


		}
	}*/

	template<unsigned int _ROWS, unsigned int _COLUMNS>
	static stm::vector<float, _ROWS> multiply256(const stm::matrix<float, _ROWS, _COLUMNS>& mat, const stm::vector<float, _COLUMNS>& vec)
	{
		constexpr unsigned int num = _COLUMNS % 64;
		constexpr unsigned int k = _COLUMNS / 64;
		constexpr unsigned int t = (k * 64);
		stm::vector<float, _ROWS> out;
		__m256 sum;
		__m256 values[8];

		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			int z = num;
			sum = _mm256_setzero_ps();
			for (unsigned int j = 0; j != k; ++j)
			{
				values[0] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 0) * 8]), load_data(&vec[((j * 8) + 0) * 8]));
				values[1] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 1) * 8]), load_data(&vec[((j * 8) + 1) * 8]));
				values[2] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 2) * 8]), load_data(&vec[((j * 8) + 2) * 8]));
				values[3] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 3) * 8]), load_data(&vec[((j * 8) + 3) * 8]));
				values[4] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 4) * 8]), load_data(&vec[((j * 8) + 4) * 8]));
				values[5] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 5) * 8]), load_data(&vec[((j * 8) + 5) * 8]));
				values[6] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 6) * 8]), load_data(&vec[((j * 8) + 6) * 8]));
				values[7] = _mm256_mul_ps(load_data(&mat[i][((j * 8) + 7) * 8]), load_data(&vec[((j * 8) + 7) * 8]));

				sum = _mm256_add_ps(sum, hadd256(values));
			}

			for (unsigned int n = 0; n < 8; ++n)
			{
				if ((z - 8) >= 0)
					values[n] = _mm256_mul_ps(load_data(&mat[i][t + (n * 8)]), load_data(&vec[t + (n * 8)]));
				else
#ifndef DEBUG
					values[n] = _mm256_mul_ps(load_data(z, &mat[i][t + (n * 8)]), load_data(z, &vec[t + (n * 8)]));
#else
					values[n] = _mm256_mul_ps(load_data(z, &mat[i][t + (n * 8)]), load_data(z, &vec[(t + (n * 8) >= _COLUMNS) ? 0 : (t + (n * 8))]));
#endif
				z -= 8;
			}
			sum = _mm256_add_ps(sum, hadd256(values));

			out[i] = addRegister(sum);
		}

		return out;
	}
}
#endif /* avx_math_h */
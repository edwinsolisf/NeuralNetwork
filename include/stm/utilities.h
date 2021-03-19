#ifndef stm_utilities_h
#define stm_utilities_h

#include "vector.h"
#include "matrix.h"
#include "dynamic_matrix.h"

namespace stm
{
	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	static void Print(const matrix<_TYPE, _ROWS, _COLUMNS>& mat);

	template<typename _TYPE>
	static void Print(const dynamic_matrix<_TYPE>& mat);

	template<typename _TYPE, unsigned int _DIM>
	static void Print(const vector<_TYPE, _DIM>& vec);

	template<typename _TYPE>
	static void Print(const dynamic_vector<_TYPE>& vec);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	inline constexpr static size_t sizeOf(const matrix<_TYPE, _ROWS, _COLUMNS>& mat) {
		return sizeof(mat);
	}

	template<typename _TYPE>
	inline static size_t sizeOf(const dynamic_matrix<_TYPE>& mat) {
		return mat.GetSize() * sizeof(_TYPE);
	}

	template<typename _TYPE, unsigned int _DIM>
	inline constexpr static size_t sizeOf(const vector<_TYPE, _DIM>& vec) {
		return sizeof(vec);
	}

	template<typename _TYPE>
	inline static size_t sizeOf(const dynamic_vector<_TYPE>& vec) {
		return vec.GetSize() * sizeof(_TYPE);
	}

}

#include "utilities.ipp"

#endif /*stm_utilities_h*/
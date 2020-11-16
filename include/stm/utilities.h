#ifndef stm_utilities_h
#define stm_utilities_h

#include "vector.h"
#include "matrix.h"
#include "dynamic_matrix.h"

namespace stm
{
	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	void Print(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		std::cout << "[ ";
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				std::cout << mat[i][j] << (((i != _ROWS - 1) || (j != _COLUMNS - 1)) ? " , " : " ");
			if (i != _ROWS - 1)
				std::cout << std::endl << "  ";
			else
				std::cout << "]" << std::endl;
		}
	}

	template<typename _TYPE>
	void Print(const dynamic_matrix<_TYPE>& mat)
	{
		std::cout << "[ ";
		for (unsigned int i = 0; i < mat.GetRowSize(); ++i)
		{
			for (unsigned int j = 0; j < mat.GetColumnSize(); ++j)
				std::cout << mat[i][j] << (((i != mat.GetRowSize() - 1) || (j != mat.GetColumnSize() - 1)) ? " , " : " ");
			if (i != mat.GetRowSize() - 1)
				std::cout << std::endl << "  ";
			else
				std::cout << "]" << std::endl;
		}
	}

	template<typename _TYPE, unsigned int _DIM>
	void Print(const vector<_TYPE, _DIM>& vec)
	{
		std::cout << "[ ";
		for (unsigned int i = 0; i < _DIM; ++i)
			std::cout << vec[i] << ((i < _DIM - 1) ? " , " : " ");
		std::cout << "]" << std::endl;
	}

	template<typename _TYPE>
	void Print(const dynamic_vector<_TYPE>& vec)
	{
		std::cout << "[ ";
		for (unsigned int i = 0; i < vec.GetSize(); ++i)
			std::cout << vec[i] << ((i < vec.GetSize() - 1) ? " , " : " ");
		std::cout << "]" << std::endl;
	}
}
#endif /*stm_utilities_h*/
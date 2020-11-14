#pragma once
#include <iostream>
#include "vector.h"

namespace stm
{
	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	class matrix
	{
	private:
		_TYPE _data[_ROWS * _COLUMNS];

	public:

		//Constructors
		matrix()
		{
			memset(_data, 0, GetSize() * sizeof(_TYPE));
		}

		matrix(const _TYPE& value)
		{
			for (unsigned int i = 0; i < GetSize(); ++i)
				_data[i] = value;
		}

		matrix(const _TYPE data[_ROWS * _COLUMNS])
		{
			memcpy(_data, data, GetSize() * sizeof(_TYPE));
		}

		matrix(const matrix& other)
		{
			memcpy(_data, other._data, GetSize() * sizeof(_TYPE));
		}

		//Unary Operators
        inline _TYPE* operator[](const unsigned int& index) { assert(index < _ROWS); return &_data[index * _COLUMNS]; }
        inline const _TYPE* operator[](const unsigned int& index) const { assert(index < _ROWS); return &_data[index * _COLUMNS]; }

		inline matrix operator+() const
		{
			return *this;
		}

		matrix operator-() const
		{
			_TYPE data[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				data[i] = -_data[i];
			return matrix(data);
		}

		//Binary Operators
		matrix operator+(const matrix& other) const
		{
			_TYPE temp[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp[i] = _data[i] + other._data[i];
			return matrix(temp);
		}

		matrix operator-(const matrix& other) const
		{
			_TYPE temp[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp[i] = _data[i] - other._data[i];
			return matrix(temp);
		}

		matrix operator*(const matrix& other) const
		{
			_TYPE temp[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp[i] = _data[i] * other._data[i];
			return matrix(temp);
		}

		matrix operator/(const matrix& other) const
		{
			_TYPE temp[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp[i] = _data[i] / other._data[i];
			return matrix(temp);
		}

		matrix operator+(const _TYPE& other) const
		{
			_TYPE temp[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp[i] = _data[i] + other;
			return matrix(temp);
		}

		matrix operator-(const _TYPE& other) const
		{
			_TYPE temp[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp[i] = _data[i] - other;
			return matrix(temp);
		}

		matrix operator*(const _TYPE& other) const
		{
			_TYPE temp[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp[i] = _data[i] * other;
			return matrix(temp);
		}

		matrix operator/(const _TYPE& other) const
		{
			_TYPE temp[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp[i] = _data[i] / other;
			return matrix(temp);
		}

		//Binary-Assigment Operators
		matrix& operator+=(const matrix& other)
		{
			*this = *this + other;
			return *this;
		}

		matrix& operator-=(const matrix& other)
		{
			*this = *this - other;
			return *this;
		}

		matrix& operator*=(const matrix& other)
		{
			*this = *this * other;
			return *this;
		}

		matrix& operator/=(const matrix& other)
		{
			*this = *this / other;
			return *this;
		}

		matrix& operator+=(const _TYPE& other)
		{
			*this = *this + other;
			return *this;
		}

		matrix& operator-=(const _TYPE& other)
		{
			*this = *this - other;
			return *this;
		}

		matrix& operator*=(const _TYPE& other)
		{
			*this = *this * other;
			return *this;
		}

		matrix& operator/=(const _TYPE& other)
		{
			*this = *this / other;
			return *this;
		}

		//Math functions
		matrix<_TYPE, _ROWS - 1, _COLUMNS - 1> Minor(const unsigned int& row, const unsigned int& column) const
		{
			_TYPE temp[(_ROWS - 1) * (_COLUMNS - 1)];
			unsigned int k = 0;

			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				if (i != row)
				{
					for (unsigned int j = 0; j < _COLUMNS; ++j)
					{
						if (j != column)
						{
							temp[k] = _data[(i * _COLUMNS) + j];
							++k;
						}
					}
				}
			}
			return matrix<_TYPE, _ROWS - 1, _COLUMNS - 1>(temp);
		}

		matrix Inverse() const
		{
			static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
			_TYPE temp[_ROWS * _COLUMNS];
			_TYPE determinant = 0;

			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				for (unsigned int j = 0; j < _COLUMNS; ++j)
				{
					temp[(i * _COLUMNS) + j] = (((i + j) % 2) ? -(this->Minor(i, j).Determinant()) : (this->Minor(i, j).Determinant()));
				}
			}

			for (unsigned int k = 0; k < _COLUMNS; ++k)
				determinant += temp[k] * _data[k];

			return matrix(temp).Transpose() / determinant;
		}

		matrix<_TYPE, _COLUMNS, _ROWS> Transpose() const
		{
			_TYPE temp[_COLUMNS * _ROWS];
			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				for (unsigned int j = 0; j < _COLUMNS; ++j)
					temp[i + (j * _ROWS)] = (*this)[i][j];
			}
			return matrix<_TYPE, _COLUMNS, _ROWS>(temp);
		}

		template<unsigned int rows, unsigned columns>
		matrix<_TYPE, rows, columns> SubMatrix(const unsigned int rowOffset, const unsigned int& columnOffset) const
		{
			_TYPE temp[rows * columns];
			for (unsigned int i = rowOffset; i < rowOffset + rows; ++i)
				memcpy(temp[i - rowOffset], _data[(i * _COLUMNS) + columnOffset], columns * sizeof(_TYPE));
			return matrix<_TYPE, rows, columns>(temp);
		}


		template<unsigned int rows>
		matrix<_TYPE, rows, _COLUMNS> Multiplication(const matrix<_TYPE, rows, _ROWS>& other) const
		{
			return matrix<_TYPE, rows, _COLUMNS>(other);
		}

		_TYPE Determinant() const
		{
			static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");

			_TYPE sum = 0;
			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				sum += (i % 2) ? -(this->Minor(0, i).Determinant() * _data[i]) : (Minor(0, i).Determinant() * _data[i]);
			}
			return sum;
		}
        
        vector<_TYPE, _ROWS> Multiply(const vector<_TYPE, _COLUMNS>& vec)
        {
            _TYPE dim[_ROWS];
            memset(dim, 0, _ROWS * sizeof(_TYPE));
            for(unsigned int i = 0; i < _ROWS; ++i)
            {
                for(unsigned int j = 0; j < _COLUMNS; ++j)
                    dim[i] += vec[j] * (*this)[i][j];
            }
            return vector<_TYPE, _ROWS>(dim);
        }
        
		//Vector Getters and Setters
		vector<_TYPE, _COLUMNS> GetRowVector(const unsigned int& row) const { return vector<_TYPE, _COLUMNS>(&_data[row * _COLUMNS]); }
		vector<_TYPE, _ROWS> GetColumnVector(const unsigned int& column) const
		{
			_TYPE temp[_ROWS];
			for (unsigned int i = 0; i < _ROWS; ++i)
				temp[i] = (*this)[i][column];
			return vector<_TYPE, _ROWS>(temp);
		}

		matrix& SetRowVector(const unsigned int& row, const vector<_TYPE, _COLUMNS>& vec)
		{
			memcpy(&_data[row * _COLUMNS], vec.GetData(), _COLUMNS * sizeof(_TYPE));
			return *this;
		}

		matrix& SetColumnVector(const unsigned int& column, const vector<_TYPE, _ROWS>& vec)
		{
			for (unsigned int i = 0; i < _ROWS; ++i)
				(*this)[i][column] = vec[i];
			return *this;
		}


		//Data manipulation functions
		matrix& ApplyToMatrix(_TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < GetSize(); ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		matrix& ApplyToRow(const unsigned int& row, _TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < _COLUMNS; ++i)
				_data[(row * _COLUMNS) + i] = func(_data[(row * _COLUMNS) + i]);
			return *this;
		}

		matrix& ApplyToColumn(const unsigned int& column, _TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < _ROWS; ++i)
				(*this)[i][column] = func((*this)[i][column]);
			return *this;
		}

		//Casting
		template<typename O_TYPE>
		matrix<O_TYPE, _ROWS, _COLUMNS> Cast() const
		{
			O_TYPE data[_ROWS * _COLUMNS];
			for (unsigned int i = 0; i < GetSize(); ++i)
				data[i] = O_TYPE(_data[i]);
			return matrix<O_TYPE, _ROWS, _COLUMNS>(data);
		}


		//Data Info Functions
		inline _TYPE* GetData() { return _data; }
		inline const _TYPE* GetData() const { return _data; }
		constexpr inline unsigned int GetRowSize() const { return _ROWS; }
		constexpr inline unsigned int GetColumnSize() const { return _COLUMNS; }
		constexpr inline unsigned int GetSize() const { return _ROWS * _COLUMNS; }
	};

	template<typename _TYPE>
	class matrix<_TYPE, 2, 2>
	{
	private:
		_TYPE _data[4];

	public:

		//Constructors
		matrix()
		{
			memset(_data, 0, GetSize() * sizeof(_TYPE));
		}

		matrix(const _TYPE& value)
		{
			for (unsigned int i = 0; i < GetSize(); ++i)
				_data[i] = value;
		}

		matrix(const _TYPE& val_00, const _TYPE& val_01, const _TYPE& val_10, const _TYPE& val_11)
		{
			_data[0] = val_00;
			_data[1] = val_01;
			_data[2] = val_10;
			_data[3] = val_11;
		}

		matrix(const _TYPE data[4])
		{
			memcpy(_data, data, GetSize() * sizeof(_TYPE));
		}

		matrix(const matrix& other)
		{
			memcpy(_data, other._data, GetSize() * sizeof(_TYPE));
		}

		//Unary Operators
        inline _TYPE* operator[](const unsigned int& index) { assert(index < 2); return &_data[index * 2]; }
        inline const _TYPE* operator[](const unsigned int& index) const { assert(index < 2); return &_data[index * 2]; }

		inline matrix operator+() const
		{
			return *this;
		}

		inline matrix operator-() const
		{
			return matrix(-_data[0], _data[1], _data[2], _data[3]);
		}

		//Binary Operators
		inline matrix operator+(const matrix& other) const
		{
			return matrix(_data[0] + other._data[0], _data[1] + other._data[1], _data[2] + other._data[2], _data[3] + other._data[3]);
		}

		inline matrix operator-(const matrix& other) const
		{
			return matrix(_data[0] - other._data[0], _data[1] - other._data[1], _data[2] - other._data[2], _data[3] - other._data[3]);
		}

		inline matrix operator*(const matrix& other) const
		{
			return matrix(_data[0] * other._data[0], _data[1] * other._data[1], _data[2] * other._data[2], _data[3] * other._data[3]);
		}

		inline matrix operator/(const matrix& other) const
		{
			return matrix(_data[0] / other._data[0], _data[1] / other._data[1], _data[2] / other._data[2], _data[3] / other._data[3]);
		}

		inline matrix operator+(const _TYPE& other) const
		{
			return matrix(_data[0] + other, _data[1] + other, _data[2] + other, _data[3] + other);
		}

		inline matrix operator-(const _TYPE& other) const
		{
			return matrix(_data[0] - other, _data[1] - other, _data[2] - other, _data[3] - other);
		}

		inline matrix operator*(const _TYPE& other) const
		{
			return matrix(_data[0] * other, _data[1] * other, _data[2] * other, _data[3] * other);
		}

		inline matrix operator/(const _TYPE& other) const
		{
			return matrix(_data[0] / other, _data[1] / other, _data[2] / other, _data[3] / other);
		}

		//Binary-Assigment Operators
		matrix& operator+=(const matrix& other)
		{
			*this = *this + other;
			return *this;
		}

		matrix& operator-=(const matrix& other)
		{
			*this = *this - other;
			return *this;
		}

		matrix& operator*=(const matrix& other)
		{
			*this = *this * other;
			return *this;
		}

		matrix& operator/=(const matrix& other)
		{
			*this = *this / other;
			return *this;
		}

		matrix& operator+=(const _TYPE& other)
		{
			*this = *this + other;
			return *this;
		}

		matrix& operator-=(const _TYPE& other)
		{
			*this = *this - other;
			return *this;
		}

		matrix& operator*=(const _TYPE& other)
		{
			*this = *this * other;
			return *this;
		}

		matrix& operator/=(const _TYPE& other)
		{
			*this = *this / other;
			return *this;
		}

		//Math functions
		inline matrix Inverse() const
		{
			return matrix(_data[3], -_data[1], -_data[2], _data[0]) / Determinant();
		}

		inline matrix Transpose() const
		{
			return matrix(_data[0], _data[2], _data[1], _data[3]);
		}

		template<unsigned int rows>
		matrix<_TYPE, rows, 2> Multiplication(const matrix<_TYPE, rows, 2>& other) const
		{
			return matrix<_TYPE, rows, 2>(other);
		}

        vector<_TYPE, 2> Multiply(const vector<_TYPE, 2>& vec)
        {
            _TYPE dim[2] = { 0, 0 };
            for(unsigned int i = 0; i < 2; ++i)
            {
                for(unsigned int j = 0; j < 2; ++j)
                    dim[i] += vec[j] * (*this)[i][j];
            }
            return vector<_TYPE, 2>(dim);
        }
        
		inline _TYPE Determinant() const
		{
			return (_data[0] * _data[3]) - (_data[1] * _data[2]);
		}

		//Vector Getters and Setters
		inline vector<_TYPE, 2> GetRowVector(const unsigned int& row) const { return vector<_TYPE, 2>(&_data[row * 2]); }
		inline vector<_TYPE, 2> GetColumnVector(const unsigned int& column) const { return vector<_TYPE, 2>((*this)[0][column], (*this)[1][column]); }

		matrix& SetRowVector(const unsigned int& row, const vector<_TYPE, 2>& vec)
		{
			memcpy(&_data[row * 2], vec.GetData(), 2 * sizeof(_TYPE));
			return *this;
		}

		matrix& SetColumnVector(const unsigned int& column, const vector<_TYPE, 2>& vec)
		{
			for (unsigned int i = 0; i < 2; ++i)
				(*this)[i][column] = vec[i];
			return *this;
		}


		//Data manipulation functions
		matrix& ApplyToMatrix(_TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < GetSize(); ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		matrix& ApplyToRow(const unsigned int& row, _TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < 2; ++i)
				_data[(row * 2) + i] = func(_data[(row * 2) + i]);
			return *this;
		}

		matrix& ApplyToColumn(const unsigned int& column, _TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < 2; ++i)
				(*this)[i][column] = func((*this)[i][column]);
			return *this;
		}

		//Casting
		template<typename O_TYPE>
		matrix<O_TYPE, 2, 2> Cast() const
		{
			O_TYPE data[4];
			for (unsigned int i = 0; i < GetSize(); ++i)
				data[i] = O_TYPE(_data[i]);
			return matrix<O_TYPE, 2, 2>(data);
		}

		//Data Info Functions
		inline _TYPE* GetData() { return _data; }
		inline const _TYPE* GetData() const { return _data; }
		constexpr inline unsigned int GetRowSize() const { return 2; }
		constexpr inline unsigned int GetColumnSize() const { return 2; }
		constexpr inline unsigned int GetSize() const { return 4; }
	};

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS, unsigned int O_COLUMNS>
	matrix<_TYPE, _ROWS, O_COLUMNS> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat1, const matrix<_TYPE, _COLUMNS, O_COLUMNS>& mat2)
	{
		_TYPE data[_ROWS * O_COLUMNS];
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < O_COLUMNS; ++j)
			{
				_TYPE sum = 0;
				for (unsigned int k = 0; k < _COLUMNS; ++k)
					sum += mat1[i][k] * mat2[k][j];
				data[(i * O_COLUMNS) + j] = sum;
			}
		}
		return matrix<_TYPE, _ROWS, O_COLUMNS>(data);
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	vector<_TYPE, _ROWS> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat, const vector<_TYPE, _COLUMNS>& vec)
	{
		_TYPE data[_ROWS];
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			_TYPE sum = 0;
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				sum += mat[i][j] * vec[j];
		}
		return vector<_TYPE, _ROWS>(data);
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> pow(const matrix<_TYPE, _ROWS, _COLUMNS>& mat, const unsigned int& power)
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		switch (power)
		{
		case 2:
			return multiply(mat, mat);
			break;
		case 3:
			return multiply(mat, multiply(mat, mat));
			break;
		default:
		{
			if (power % 2)
				return multiply(mat, multiply(pow(mat, power / 2), pow(mat, power / 2)));
			else
				return multiply(pow(mat, power / 2), pow(mat, power / 2));
			break;
		}
		}
	}

	template<typename _TYPE>
	matrix<_TYPE, 2, 2> sqrt(const matrix<_TYPE, 2, 2>& mat)
	{
		_TYPE temp = sqrt(determinant(mat));
		return matrix<_TYPE, 2, 2>(mat[0][0] + temp, mat[0][1], mat[1][0], mat[1][1] + temp) / (sqrt(temp + 2 * (mat[0][0] + mat[1][1])));
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	_TYPE determinant(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");

		_TYPE sum = 0;
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			sum += (i % 2) ? -(determinant(mat.Minor(0, i)) * mat[0][i]) : (determinant(mat.Minor(0, i)) * mat[0][i]);
		}
		return sum;
	}

	template<typename _TYPE>
	_TYPE determinant(const matrix<_TYPE, 2, 2>& mat)
	{
		return (mat[0][0] * mat[1][1]) - (mat[0][1] * mat[1][0]);
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> inverse(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		_TYPE temp[_ROWS * _COLUMNS];
		_TYPE det = 0;

		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
			{
				temp[(i * _COLUMNS) + j] = (((i + j) % 2) ? -(determinant(mat.Minor(i, j))) : (determinant(mat.Minor(i, j))));
			}
		}

		for (unsigned int k = 0; k < _COLUMNS; ++k)
			det += temp[k] * mat[0][k];

		return matrix<_TYPE, _ROWS, _COLUMNS>(temp).Transpose() / det;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> cofactorMatrix(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		_TYPE temp[_ROWS * _COLUMNS];

		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
			{
				temp[(i * _COLUMNS) + j] = (((i + j) % 2) ? -(determinant(mat.Minor(i, j))) : (determinant(mat.Minor(i, j))));
			}
		}

		return matrix<_TYPE, _ROWS, _COLUMNS>(temp);
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	inline matrix<_TYPE, _ROWS, _COLUMNS> adjugate(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		return cofactorMatrix(mat).Transpose();
	}

	template<typename _TYPE, unsigned int _DIM>
	inline matrix<_TYPE, 1, _DIM> torowVector(const vector<_TYPE, _DIM>& vec)
	{
		return matrix<_TYPE, 1, _DIM>(vec.GetData());
	}

	template<typename _TYPE, unsigned int _DIM>
	inline matrix<_TYPE, _DIM, 1> tocolumnVector(const vector<_TYPE, _DIM>& vec)
	{
		return matrix<_TYPE, _DIM, 1>(vec.GetData());
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	constexpr matrix<_TYPE, _ROWS, _COLUMNS> GetIndentityMatrix()
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		matrix<_TYPE, _ROWS, _COLUMNS> mat;
		for (unsigned int i = 0; i < _ROWS; ++i)
			mat[i][i] = (_TYPE)1;
		return mat;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	constexpr matrix<_TYPE, _ROWS, _COLUMNS> GetZeroMatrix()
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		return matrix<_TYPE, _ROWS, _COLUMNS>();
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	constexpr matrix<_TYPE, _ROWS, _COLUMNS> GetExchangeMatrix()
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		matrix<_TYPE, _ROWS, _COLUMNS> mat;
		for (unsigned int i = 0; i < _ROWS; ++i)
			mat[i][_ROWS - 1 - i] = (_TYPE)1;
		return mat;
	}

	typedef matrix<int, 2, 2> mat2i;
	typedef matrix<float, 2, 2> mat2f;
	typedef matrix<int, 3, 3> mat3i;
	typedef matrix<float, 3, 3> mat3f;
	typedef matrix<int, 4, 4> mat4i;
	typedef matrix<float, 4, 4> mat4f;

	const mat2i identity_mat2i = GetIndentityMatrix<int, 2, 2>();
	const mat2f identity_mat2f = GetIndentityMatrix<float, 2, 2>();
	const mat3i identity_mat3i = GetIndentityMatrix<int, 3, 3>();
	const mat3f identity_mat3f = GetIndentityMatrix<float, 3, 3>();
	const mat4i identity_mat4i = GetIndentityMatrix<int, 4, 4>();
	const mat4f identity_mat4f = GetIndentityMatrix<float, 4, 4>();

}

#ifndef stm_matrix_h
#define stm_matrix_h

#include <iostream>
#include "vector.h"
#include "dynamic_vector.h"

namespace stm
{
	template<typename _TYPE>
	class dynamic_matrix;

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
			std::fill_n(_data, GetSize(), value);
		}

		matrix(const _TYPE data[_ROWS * _COLUMNS])
		{
			memcpy(_data, data, GetSize() * sizeof(_TYPE));
		}

		matrix(const matrix& other)
		{
			memcpy(_data, other._data, GetSize() * sizeof(_TYPE));
		}

		//Assigment Operators
		matrix& operator=(const dynamic_matrix<_TYPE>& mat)
		{
			stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
			memcpy(_data, mat.GetData(), GetSize() * sizeof(_TYPE));
			return *this;
		}

		//Unary Operators
        inline _TYPE* operator[](unsigned int index) { stm_assert(index < _ROWS); return &_data[index * _COLUMNS]; }
        inline const _TYPE* operator[](unsigned int index) const { stm_assert(index < _ROWS); return &_data[index * _COLUMNS]; }

		inline matrix operator+() const
		{
			return *this;
		}

		matrix operator-() const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = -_data[i];
			return temp;
		}

		//Binary Operators
		matrix operator+(const matrix& other) const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] + other._data[i];
			return temp;
		}

		matrix operator-(const matrix& other) const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] - other._data[i];
			return temp;
		}

		matrix operator*(const matrix& other) const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] * other._data[i];
			return matrix(temp);
		}

		matrix operator/(const matrix& other) const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] / other._data[i];
			return temp;
		}

		matrix operator+(const _TYPE& other) const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] + other;
			return temp;
		}

		matrix operator-(const _TYPE& other) const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] - other;
			return temp;
		}

		matrix operator*(const _TYPE& other) const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] * other;
			return temp;
		}

		matrix operator/(const _TYPE& other) const
		{
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] / other;
			return temp;
		}

		matrix operator+(const dynamic_matrix<_TYPE>& mat) const
		{
			stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] + mat.GetData()[i];
			return temp;
		}

		matrix operator-(const dynamic_matrix<_TYPE>& mat) const
		{
			stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] - mat.GetData()[i];
			return temp;
		}

		matrix operator*(const dynamic_matrix<_TYPE>& mat) const
		{
			stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] * mat.GetData()[i];
			return temp;
		}

		matrix operator/(const dynamic_matrix<_TYPE>& mat) const
		{
			stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
			matrix temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = _data[i] / mat.GetData()[i];
			return temp;
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

		matrix& operator+=(const dynamic_matrix<_TYPE>& mat)
		{
			*this = *this + mat;
			return *this;
		}

		matrix& operator-=(const dynamic_matrix<_TYPE>& mat)
		{
			*this = *this - mat;
			return *this;
		}

		matrix& operator*=(const dynamic_matrix<_TYPE>& mat)
		{
			*this = *this * mat;
			return *this;
		}

		matrix& operator/=(const dynamic_matrix<_TYPE>& mat)
		{
			*this = *this / mat;
			return *this;
		}

		//Math functions
		matrix<_TYPE, _ROWS - 1, _COLUMNS - 1> Minor(unsigned int row, unsigned int column) const
		{
			matrix<_TYPE, _ROWS - 1, _COLUMNS - 1> temp;
			unsigned int k = 0;

			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				if (i != row)
				{
					for (unsigned int j = 0; j < _COLUMNS; ++j)
					{
						if (j != column)
						{
							temp._data[k] = _data[(i * _COLUMNS) + j];
							++k;
						}
					}
				}
			}
			return temp;
		}

		matrix Inverse() const
		{
			static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
			matrix temp;
			_TYPE determinant = 0;

			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				for (unsigned int j = 0; j < _COLUMNS; ++j)
				{
					temp._data[(i * _COLUMNS) + j] = (((i + j) % 2) ? -(this->Minor(i, j).Determinant()) : (this->Minor(i, j).Determinant()));
				}
			}

			for (unsigned int k = 0; k < _COLUMNS; ++k)
				determinant += temp._data[k] * _data[k];

			return temp.Transpose() / determinant;
		}

		matrix<_TYPE, _COLUMNS, _ROWS> Transpose() const
		{
			matrix<_TYPE, _COLUMNS, _ROWS> temp;
			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				for (unsigned int j = 0; j < _COLUMNS; ++j)
					temp[0][i + (j * _ROWS)] = (*this)[i][j];
			}
			return temp;
		}

		template<unsigned int rows, unsigned columns>
		matrix<_TYPE, rows, columns> SubMatrix(unsigned int rowOffset, unsigned int columnOffset) const
		{
			matrix temp;
			for (unsigned int i = rowOffset; i < rowOffset + rows; ++i)
				memcpy(temp._data[i - rowOffset], _data[(i * _COLUMNS) + columnOffset], columns * sizeof(_TYPE));
			return temp;
		}

		_TYPE Determinant() const
		{
			static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");

			_TYPE sum = 0;
			for (unsigned int i = 0; i < _ROWS; ++i)
				sum += (i % 2) ? -(Minor(0, i).Determinant() * _data[i]) : (Minor(0, i).Determinant() * _data[i]);
			return sum;
		}
        
		template<unsigned int O_COLUMNS>
		matrix<_TYPE, _ROWS, O_COLUMNS> Multiply(const matrix<_TYPE, _COLUMNS, O_COLUMNS>& mat) const
		{
			matrix<_TYPE, _ROWS, O_COLUMNS> temp;
			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				for (unsigned int j = 0; j < O_COLUMNS; ++j)
				{
					_TYPE sum = 0;
					for (unsigned int k = 0; k < _COLUMNS; ++k)
						sum += (*this)[i][k] * mat[k][j];
					temp._data[(i * O_COLUMNS) + j] = sum;
				}
			}
			return temp;
		}

        vector<_TYPE, _ROWS> Multiply(const vector<_TYPE, _COLUMNS>& vec) const
        {
			vector<_TYPE, _ROWS> temp;
            for(unsigned int i = 0; i < _ROWS; ++i)
            {
                for(unsigned int j = 0; j < _COLUMNS; ++j)
                    temp[i] += vec[j] * (*this)[i][j];
            }
			return temp;
        }
        
		vector<_TYPE, _ROWS> Multiply(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(_COLUMNS == vec.GetSize());
			vector<_TYPE, _ROWS> temp;
			for (unsigned int i = 0; i < _ROWS; ++i)
			{
				for (unsigned int j = 0; j < _COLUMNS; ++j)
					temp[i] += vec[j] * (*this)[i][j];
			}
			return temp;
		}

		//Vector Getters and Setters
		vector<_TYPE, _COLUMNS> GetRowVector(unsigned int row) const { return vector<_TYPE, _COLUMNS>(&_data[row * _COLUMNS]); }
		vector<_TYPE, _ROWS> GetColumnVector(unsigned int column) const
		{
			vector<_TYPE, _ROWS> temp;
			for (unsigned int i = 0; i < _ROWS; ++i)
				temp[i] = (*this)[i][column];
			return temp;
		}

		matrix& SetRowVector(unsigned int row, const vector<_TYPE, _COLUMNS>& vec)
		{
			memcpy(&_data[row * _COLUMNS], vec.GetData(), _COLUMNS * sizeof(_TYPE));
			return *this;
		}

		matrix& SetRowVector(unsigned int row, const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(_COLUMNS == vec.GetSize());
			memcpy(&_data[row * _COLUMNS], vec.GetData(), _COLUMNS * sizeof(_TYPE));
			return *this;
		}

		matrix& SetColumnVector(unsigned int column, const vector<_TYPE, _ROWS>& vec)
		{
			for (unsigned int i = 0; i < _ROWS; ++i)
				(*this)[i][column] = vec[i];
			return *this;
		}

		matrix& SetColumnVector(unsigned int column, const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(_ROWS == vec.GetSize());
			for (unsigned int i = 0; i < _ROWS; ++i)
				(*this)[i][column] = vec[i];
			return *this;
		}

		//Data manipulation functions
		matrix& ApplyToMatrix(_TYPE(*func)(_TYPE))
		{
			for (unsigned int i = 0; i < GetSize(); ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		matrix& ApplyToRow(unsigned int row, _TYPE(*func)(_TYPE))
		{
			for (unsigned int i = 0; i < _COLUMNS; ++i)
				_data[(row * _COLUMNS) + i] = func(_data[(row * _COLUMNS) + i]);
			return *this;
		}

		matrix& ApplyToColumn(unsigned int column, _TYPE(*func)(_TYPE))
		{
			for (unsigned int i = 0; i < _ROWS; ++i)
				(*this)[i][column] = func((*this)[i][column]);
			return *this;
		}

		matrix& SetAll(_TYPE value)
		{
			for (unsigned int i = 0; i < GetSize(); ++i)
				_data[i] = value;
			return *this;
		}

		//Casting
		template<typename O_TYPE>
		matrix<O_TYPE, _ROWS, _COLUMNS> Cast() const
		{
			matrix<O_TYPE, _ROWS, _COLUMNS> temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = O_TYPE(_data[i]);
			return temp;
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
			std::fill_n(_data, GetSize(), value);
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

		//Assigment Operator
		matrix& operator=(const dynamic_matrix<_TYPE>& mat)
		{
			stm_assert(mat.GetColumnSize() == 2 && mat.GetRowSize() == 2);
			memcpy(_data, mat.GetData(), 4 * sizeof(_TYPE));
			return *this;
		}

		//Unary Operators
        inline _TYPE* operator[](unsigned int index) { stm_assert(index < 2); return &_data[index * 2]; }
        inline const _TYPE* operator[](unsigned int index) const { stm_assert(index < 2); return &_data[index * 2]; }

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

		inline matrix operator+(const dynamic_matrix<_TYPE>& mat) const
		{
			stm_assert(mat.GetColumnSize() == 2 && mat.GetRowSize() == 2);
			return matrix(_data[0] + mat.GetData[0], _data[1] + mat.GetData()[1], _data[2] + mat.GetData()[2], _data[3] + mat.GetData()[3]);
		}

		inline matrix operator-(const dynamic_matrix<_TYPE>& mat) const
		{
			stm_assert(mat.GetColumnSize() == 2 && mat.GetRowSize() == 2);
			return matrix(_data[0] - mat.GetData[0], _data[1] - mat.GetData()[1], _data[2] - mat.GetData()[2], _data[3] - mat.GetData()[3]);
		}

		inline matrix operator*(const dynamic_matrix<_TYPE>& mat) const
		{
			stm_assert(mat.GetColumnSize() == 2 && mat.GetRowSize() == 2);
			return matrix(_data[0] * mat.GetData[0], _data[1] * mat.GetData()[1], _data[2] * mat.GetData()[2], _data[3] * mat.GetData()[3]);
		}

		inline matrix operator/(const dynamic_matrix<_TYPE>& mat) const
		{
			stm_assert(mat.GetColumnSize() == 2 && mat.GetRowSize() == 2);
			return matrix(_data[0] / mat.GetData[0], _data[1] / mat.GetData()[1], _data[2] / mat.GetData()[2], _data[3] / mat.GetData()[3]);
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

		matrix& operator+=(const dynamic_matrix<_TYPE>& mat)
		{
			*this = *this + mat;
			return *this;
		}

		matrix& operator-=(const dynamic_matrix<_TYPE>& mat)
		{
			*this = *this - mat;
			return *this;
		}

		matrix& operator*=(const dynamic_matrix<_TYPE>& mat)
		{
			*this = *this * mat;
			return *this;
		}

		matrix& operator/=(const dynamic_matrix<_TYPE>& mat)
		{
			*this = *this / mat;
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

		template<unsigned int O_COLUMNS>
		matrix<_TYPE, 2, O_COLUMNS> Multiply(const matrix<_TYPE, 2, O_COLUMNS>& mat) const
		{
			matrix<_TYPE, 2, O_COLUMNS> temp;
			for (unsigned int i = 0; i < 2; ++i)
			{
				for (unsigned int j = 0; j < O_COLUMNS; ++j)
				{
					_TYPE sum = 0;
					for (unsigned int k = 0; k < 2; ++k)
						sum += (*this)[i][k] * mat[k][j];
					temp._data[(i * O_COLUMNS) + j] = sum;
				}
			}
			return temp;
		}

        vector<_TYPE, 2> Multiply(const vector<_TYPE, 2>& vec)
        {
			vector<_TYPE, 2> temp;
            for(unsigned int i = 0; i < 2; ++i)
            {
                for(unsigned int j = 0; j < 2; ++j)
                    temp[i] += vec[j] * (*this)[i][j];
            }
			return temp;
        }
        
		vector<_TYPE, 2> Multiply(const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(vec.GetSize() == 2);
			vector<_TYPE, 2> temp;
			for (unsigned int i = 0; i < 2; ++i)
			{
				for (unsigned int j = 0; j < 2; ++j)
					temp[i] += vec[j] * (*this)[i][j];
			}
			return temp;
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

		matrix& SetRowVector(unsigned int row, const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(vec.GetSize() == 2);
			memcpy(&_data[row * 2], vec.GetData(), 2 * sizeof(_TYPE));
			return *this;
		}

		matrix& SetColumnVector(const unsigned int& column, const vector<_TYPE, 2>& vec)
		{
			for (unsigned int i = 0; i < 2; ++i)
				(*this)[i][column] = vec[i];
			return *this;
		}

		matrix& SetColumnVector(unsigned int column, const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(vec.GetSize() == 2);
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
			matrix<O_TYPE, 2, 2> temp;
			for (unsigned int i = 0; i < GetSize(); ++i)
				temp._data[i] = O_TYPE(_data[i]);
			return temp;
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
		matrix<_TYPE, _ROWS, O_COLUMNS> temp;
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < O_COLUMNS; ++j)
			{
				_TYPE sum = 0;
				for (unsigned int k = 0; k < _COLUMNS; ++k)
					sum += mat1[i][k] * mat2[k][j];
				temp[0][(i * O_COLUMNS) + j] = sum;
			}
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	vector<_TYPE, _ROWS> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat, const vector<_TYPE, _COLUMNS>& vec)
	{
		vector<_TYPE, _ROWS> temp;
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			_TYPE sum = 0;
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				sum += mat[i][j] * vec[j];
			temp[i] = sum;
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	vector<_TYPE, _ROWS> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat, const dynamic_vector<_TYPE>& vec)
	{
		stm_assert(_COLUMNS == vec.GetSize());
		vector<_TYPE, _ROWS> temp;
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			_TYPE sum = 0;
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				sum += mat[i][j] * vec[j];
			temp = sum;
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> pow(const matrix<_TYPE, _ROWS, _COLUMNS>& mat, unsigned int power)
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
	matrix<_TYPE, _COLUMNS, _ROWS> transpose(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		matrix<_TYPE, _COLUMNS, _ROWS> temp;
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				temp[0][i + (j * _ROWS)] = (*this)[i][j];
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	_TYPE determinant(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");

		_TYPE sum = 0;
		for (unsigned int i = 0; i < _ROWS; ++i)
			sum += (i % 2) ? -(determinant(mat.Minor(0, i)) * mat[0][i]) : (determinant(mat.Minor(0, i)) * mat[0][i]);
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
		matrix<_TYPE, _ROWS, _COLUMNS> temp;
		_TYPE det = 0;

		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				temp[0][(i * _COLUMNS) + j] = (((i + j) % 2) ? -(determinant(mat.Minor(i, j))) : (determinant(mat.Minor(i, j))));
		}

		for (unsigned int k = 0; k < _COLUMNS; ++k)
			det += temp[0][k] * mat[0][k];

		return transpose(temp) / det;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> cofactorMatrix(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		matrix<_TYPE, _ROWS, _COLUMNS> temp;

		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				temp[0][(i * _COLUMNS) + j] = (((i + j) % 2) ? -(determinant(mat.Minor(i, j))) : (determinant(mat.Minor(i, j))));
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	inline matrix<_TYPE, _ROWS, _COLUMNS> adjugate(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		return transpose(cofactorMatrix(mat));
	}

	template<typename _TYPE, unsigned int _DIM>
	inline matrix<_TYPE, 1, _DIM> toRowMatrix(const vector<_TYPE, _DIM>& vec)
	{
		return matrix<_TYPE, 1, _DIM>(vec.GetData());
	}

	template<typename _TYPE, unsigned int _DIM>
	inline matrix<_TYPE, _DIM, 1> toColumnMatrix(const vector<_TYPE, _DIM>& vec)
	{
		return matrix<_TYPE, _DIM, 1>(vec.GetData());
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	inline vector<_TYPE, _ROWS * _COLUMNS> toRowVector(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		return vector<_TYPE, _ROWS * _COLUMNS>(mat.GetData());
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	inline vector<_TYPE, _ROWS * _COLUMNS> toColumnVector(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		return vector<_TYPE, _ROWS * _COLUMNS>(mat.Transpose().GetData());
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
#endif /* stm_matrix_h */
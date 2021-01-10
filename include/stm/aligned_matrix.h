#ifndef stm_aligned_matrix_h
#define stm_aligned_matrix_h

#include "debug.h"
#include "dynamic_matrix.h"

namespace stm
{
	template<typename _T>
	class aligned_matrix
	{
	public:
		aligned_matrix(unsigned int rows, unsigned int columns)
			:_rows(rows), _columns(columns), _data(new _T[getGoodSize()])
		{
			memset(_data, 0, getGoodSize() * sizeof(_T));
		}

		aligned_matrix(unsigned int rows, unsigned int columns, _T value)
			:_rows(rows), _columns(columns), _data(new _T[getGoodSize()])
		{
			memset(_data, 0, getGoodSize() * sizeof(_T));
			for (unsigned int i = 0; i < _rows; ++i)
			{
				for (unsigned int j = 0; j < _columns; ++j)
					(*this)[i][j] = value;
			}
		}

		aligned_matrix(unsigned int rows, unsigned int columns, _T* values)
			:_rows(rows), _columns(columns), _data(new _T[getGoodSize()])
		{
			memset(_data, 0, getGoodSize() * sizeof(_T));
			for (unsigned int i = 0; i < _rows; ++i)
			{
				for (unsigned int j = 0; j < _columns; ++j)
					(*this)[i][j] = values[i*_columns + j];
			}
		}

		aligned_matrix(const aligned_matrix& other)
			:_rows(other._rows), _columns(other._columns), _data(new _T[other.getGoodSize()])
		{
			memcpy(_data, other._data, other.getGoodSize() * sizeof(_T));
		}

		aligned_matrix(aligned_matrix&& other) noexcept
			:_rows(std::exchange(other._rows, 0)), _columns(std::exchange(other._columns, 0)), _data(std::exchange(other._data, nullptr))
		{
		}

		~aligned_matrix()
		{
			delete _data;
		}

		aligned_matrix operator+(const aligned_matrix& other) const
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			aligned_matrix temp(_rows, _columns);
			for (unsigned int i = 0; i < getGoodSize(); ++i)
				temp._data[i] = _data[i] + other._data[i];
			return temp;
		}

		inline _T* operator[](unsigned int index) { return &_data[index * getGoodColumnCount()]; }
		inline const _T* operator[](unsigned int index) const { return &_data[index * getGoodColumnCount()]; }

		inline unsigned int getRowCount() const { return _rows; }
		inline unsigned int getColumnCount() const { return _columns; }
		inline unsigned int getSize() const { return _rows * _columns; }

		inline unsigned int getGoodRowCount() const { return ((_rows - 1) / 4 + 1) * 4; }
		inline unsigned int getGoodColumnCount() const { return ((_columns - 1) / 4 + 1) * 4; }
		inline unsigned int getGoodSize() const { return getGoodRowCount() * getGoodColumnCount(); }

		inline _T* getData() { return _data; }
		inline const _T* getData() const { return _data; }

		dynamic_matrix<_T> ToDynamic() const
		{
			dynamic_matrix<_T> temp(_rows, _columns);
			for (unsigned int i = 0; i < _rows; ++i)
				for (unsigned int j = 0; j < _columns; ++j)
					temp[i][j] = (*this)[i][j];
			return temp;
		}

		aligned_matrix Multiply(const aligned_matrix& mat) const
		{
			stm_assert(_columns == mat._rows);
			aligned_matrix temp(_rows, mat._columns);
			for (unsigned int i = 0; i < _rows; ++i)
			{
				for (unsigned int j = 0; j < mat._columns; ++j)
				{
					_T sum = 0;
					for (unsigned int k = 0; k < _columns; ++k)
						sum += (*this)[i][k] * mat[k][j];
					temp[i][j] = sum;
				}
			}
			return temp;
		}

		aligned_matrix mult(const aligned_matrix& mat) const
		{
			stm_assert(_columns == mat._rows);
			aligned_matrix temp(_rows, mat._columns);
			for (unsigned int i = 0; i < _rows; ++i)
			{
				for (unsigned int j = 0; j < _columns; ++j)
				{
					for (unsigned int k = 0; k < mat._columns; ++k)
						temp[i][k] += (*this)[i][j] * mat[j][k];
				}
			}
			return temp;
		}

	private:
		unsigned int _rows, _columns;
		_T* _data;
	};
}

#endif /* stm_aligned_matrix_h */

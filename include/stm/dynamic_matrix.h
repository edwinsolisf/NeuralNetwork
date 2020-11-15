#ifndef dynamic_matrix_h
#define dynamic_matrix_h

#include "debug.h"
#include "matrix.h"

namespace stm
{
	template<typename _T>
	class dynamic_matrix
	{
	public:

		//Constructors
		dynamic_matrix(unsigned int rows, unsigned int columns)
			:_rows(rows), _columns(columns)
		{
			stm_assert(rows != 0 && columns != 0);
			_data = new _T[_rows * _columns]{ 0 };
		}

		dynamic_matrix(unsigned int rows, unsigned int columns, const _T* data)
			:_rows(rows), _columns(columns)
		{
			stm_assert(rows != 0 && columns != 0);
			_data = new _T[_rows * _columns];
			memcpy(_data, data, _rows * _columns * sizeof(_T));
		}

		dynamic_matrix(unsigned int rows, unsigned int columns, const _T& value)
			:_rows(rows), _columns(columns)
		{
			stm_assert(rows != 0 && columns != 0);
			_data = new _T[_rows * _columns]{ 0 };
			for (unsigned int i = 0; i < _columns * _rows; ++i)
				_data[i] = value;
		}

		dynamic_matrix(const dynamic_matrix& other)
			:_rows(other._rows), _columns(other._columns)
		{
			_data = new _T[_rows * _columns];
			memcpy(_data, other._data, _rows * _columns * sizeof(_T));
		}
		
		
		dynamic_matrix(dynamic_matrix&& other) noexcept
			:_data(std::exchange(other._data, nullptr)), _rows(std::exchange(other._rows, 0)), _columns(std::exchange(other._columns, 0))
		{
		}
		
		template<unsigned int rows, unsigned int columns>
		dynamic_matrix(const matrix<_T, rows, columns>& static_matrix)
			:_data(new _T[rows * columns]), _rows(rows), _columns(columns)
		{
			memcpy(_data, static_matrix.GetData(), _rows * _columns * sizeof(_T));
		}
		
		//Assigment Operators
		dynamic_matrix& operator=(const dynamic_matrix& other)
		{
			if (this == &other) { return *this; }
			if ((_rows * _columns) == other.GetSize())
				memcpy(_data, other._data, _rows * _columns * sizeof(_T));
			else
			{
				_T* newData = new _T[other.GetSize()];
				memcpy(newData, other._data, other.GetSize() * sizeof(_T));
				delete _data;
				_data = newData;
			}
			_rows = other._rows;
			_columns = other._columns;

			return *this;
		}
			
		dynamic_matrix& operator=(dynamic_matrix&& other) noexcept
		{	
			if (this == &other) { return *this; }
			std::swap(_data, other._data);
			std::swap(_rows, other._rows);
			std::swap(_columns, other._columns);
			
			return *this;
		}
		
		//Unary Operators
		inline _T* operator[](unsigned int index) { stm_assert(index < _rows); return &_data[index]; }
		inline const _T* operator[](unsigned int index) const { stm_assert(index < _rows); return &_data[index]; }

		inline dynamic_matrix operator+() const { return *this; }

		dynamic_matrix operator-() const
		{
			dynamic_matrix out(_rows, _columns, _data);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = -_data[i];
			return std::move(out);
		}

		//Binary Operators
		dynamic_matrix operator+(const dynamic_matrix& other) const
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] + other._data[i];
			return std::move(out);
		}

		dynamic_matrix operator-(const dynamic_matrix& other) const
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] - other._data[i];
			return std::move(out);
		}

		dynamic_matrix operator*(const dynamic_matrix& other) const
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] * other._data[i];
			return std::move(out);
		}

		dynamic_matrix operator/(const dynamic_matrix& other) const
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] / other._data[i];
			return std::move(out);
		}

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix operator+(const matrix<_T, rows, columns>& static_matrix) const
		{
			stm_assert(_rows == rows && _columns == columns);
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] + static_matrix.GetData()[i];
			return std::move(out);
		}

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix operator-(const matrix<_T, rows, columns>& static_matrix) const
		{
			stm_assert(_rows == rows && _columns == columns);
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] - static_matrix.GetData()[i];
			return std::move(out);
		}

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix operator*(const matrix<_T, rows, columns>& static_matrix) const
		{
			stm_assert(_rows == rows && _columns == columns);
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] * static_matrix.GetData()[i];
			return std::move(out);
		}

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix operator/(const matrix<_T, rows, columns>& static_matrix) const
		{
			stm_assert(_rows == rows && _columns == columns);
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] / static_matrix.GetData()[i];
			return std::move(out);
		}

		dynamic_matrix operator+(const _T& value) const
		{
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] + value;
			return std::move(out);
		}

		dynamic_matrix operator-(const _T& value) const
		{
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] - value;
			return std::move(out);
		}

		dynamic_matrix operator*(const _T& value) const
		{
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] * value;
			return std::move(out);
		}

		dynamic_matrix operator/(const _T& value) const
		{
			dynamic_matrix out(_rows, _columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				out._data[i] = _data[i] / value;
			return std::move(out);
		}

		//Binary-assigment operators
		dynamic_matrix& operator+=(const dynamic_matrix& other)
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] + other._data[i];
			return *this;
		}

		dynamic_matrix& operator-=(const dynamic_matrix& other)
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] - other._data[i];
			return *this;
		}

		dynamic_matrix& operator*=(const dynamic_matrix& other)
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] * other._data[i];
			return *this;
		}

		dynamic_matrix& operator/=(const dynamic_matrix& other)
		{
			stm_assert(_rows == other._rows && _columns == other._columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] / other._data[i];
			return *this;
		}
		
		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator+=(const matrix<_T, rows, columns>& static_matrix)
		{
			stm_assert(_rows == rows && _columns == columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] + static_matrix.GetData()[i];
			return *this;
		}

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator-=(const matrix<_T, rows, columns>& static_matrix)
		{
			stm_assert(_rows == rows && _columns == columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] - static_matrix.GetData()[i];
			return *this;
		}

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator*=(const matrix<_T, rows, columns>& static_matrix)
		{
			stm_assert(_rows == rows && _columns == columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] * static_matrix.GetData()[i];
			return *this;
		}

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator/=(const matrix<_T, rows, columns>& static_matrix)
		{
			stm_assert(_rows == rows && _columns == columns);
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] / static_matrix.GetData()[i];
			return *this;
		}

		dynamic_matrix& operator+=(const _T& value)
		{
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] + value;
			return *this;
		}

		dynamic_matrix& operator-=(const _T& value)
		{
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] - value;
			return *this;
		}

		dynamic_matrix& operator*=(const _T& value)
		{
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] * value;
			return *this;
		}

		dynamic_matrix& operator/=(const _T& value)
		{
			for (unsigned int i = 0; i < _rows * _columns; ++i)
				_data[i] = _data[i] / value;
			return *this;
		}

		dynamic_matrix Minor(unsigned int row, unsigned int column)
		{
			stm_assert(row < _rows && column < _columns);
			dynamic_matrix out(_rows - 1, _columns - 1);
			for (unsigned int i = 0; i < _rows; ++i)
			{
				if (i != row)
				{
					for (unsigned int j = 0; j < _columns; ++j)
					{
						if (j != column)
						{
							out[k] = _data[(i * _columns) + j];
							++k;
						}
					}
				}
			}
			return std::move(out);
		}

		void Resize(unsigned int rows, unsigned int columns)
		{
			stm_assert(rows != 0 && columns != 0);
			_T* newData = new _T[rows * columns];
			memcpy(newData, _data, rows * columns * sizeof(_T));
			delete[] _data;
			_data = newData;
		}

		inline unsigned int GetRows() const { return _rows; }
		inline unsigned int GetColumns() const { return _columns; }
		inline unsigned int GetSize() const { return _rows * _columns; }
		inline _T* GetData() { return _data; }
		inline const _T* GetData() const { return _data; }


		~dynamic_matrix()
		{
			delete[] _data;
		}

	private:
		_T* _data;
		unsigned int _rows, _columns;
	};

	typedef dynamic_matrix<int> mat_i;
	typedef dynamic_matrix<float> mat_f;
}
#endif /* stm_dynamic_matrix_h */
namespace stm
{
	template<typename _T>
	dynamic_matrix<_T>::dynamic_matrix(unsigned int rows, unsigned int columns)
		:_data(new _T[rows * columns]), _rows(rows), _columns(columns)
	{
		memset(_data, 0, rows * columns * sizeof(_T));
		stm_assert(rows != 0 && columns != 0);
	}

	template<typename _T>
	dynamic_matrix<_T>::dynamic_matrix(unsigned int rows, unsigned int columns, const _T* data)
		:_data(new _T[rows * columns]), _rows(rows), _columns(columns)
	{
		stm_assert(rows != 0 && columns != 0);
		memcpy(_data, data, _rows * _columns * sizeof(_T));
	}

	template<typename _T>
	dynamic_matrix<_T>::dynamic_matrix(unsigned int rows, unsigned int columns, _T value)
		:_data(new _T[rows * columns]), _rows(rows), _columns(columns)
	{
		stm_assert(rows != 0 && columns != 0);
		std::fill_n(_data, _rows * _columns, value);
	}

	template<typename _T>
	dynamic_matrix<_T>::dynamic_matrix(unsigned int rows, unsigned int columns, std::initializer_list<_T> list)
		:_data(new _T[rows * columns]), _rows(rows), _columns(columns)
	{
		stm_assert(rows != 0 && columns != 0 && list.size() == (rows * columns));
		std::copy(list.begin(), list.end(), _data);
	}

	template<typename _T>
	dynamic_matrix<_T>::dynamic_matrix(unsigned int rows, unsigned int columns, std::initializer_list<std::initializer_list<_T>> list)
		:_data(new _T[rows * columns]), _rows(rows), _columns(columns)
	{
		stm_assert(rows != 0 && columns != 0 && list.size() == rows && list.begin()->size() == columns);
		for (unsigned int i = 0; i < rows; ++i)
			std::copy(list.begin()[i].begin(), list.begin()[i].end(), &_data[i * columns]);
	}

	template<typename _T>
	dynamic_matrix<_T>::dynamic_matrix(const dynamic_matrix& other)
		:_data(new _T[other._rows * other._columns]), _rows(other._rows), _columns(other._columns)
	{
		memcpy(_data, other._data, _rows * _columns * sizeof(_T));
	}

	template<typename _T>
	dynamic_matrix<_T>::~dynamic_matrix()
	{
		delete[] _data;
	}

	template<typename _T>
	dynamic_matrix<_T>::dynamic_matrix(dynamic_matrix&& other) noexcept
		:_data(std::exchange(other._data, nullptr)), _rows(std::exchange(other._rows, 0)), _columns(std::exchange(other._columns, 0))
	{
	}

	template<typename _T>
	dynamic_matrix<_T>::dynamic_matrix(_T*& data, unsigned int rows, unsigned int columns)
		:_data(std::exchange(data, nullptr)), _rows(rows), _columns(columns)
	{
		stm_assert(_data != nullptr);
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	dynamic_matrix<_T>::dynamic_matrix(const matrix<_T, rows, columns>& static_matrix)
		: _data(new _T[rows * columns]), _rows(rows), _columns(columns)
	{
		memcpy(_data, static_matrix.GetData(), _rows * _columns * sizeof(_T));
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator=(const dynamic_matrix& other)
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

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator=(dynamic_matrix&& other) noexcept
	{
		if (this == &other) { return *this; }
		std::swap(_data, other._data);
		std::swap(_rows, other._rows);
		std::swap(_columns, other._columns);

		return *this;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator=(const matrix<_T, rows, columns>& mat)
	{
		stm_assert(_rows == rows && _columns == _columns);
		memcpy(_data, mat.GetData(), _rows * _columns * sizeof(_T));
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator-() const&
	{
		dynamic_matrix temp(_rows, _columns, _data);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = -_data[i];
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator-() &&
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = -_data[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator+(const dynamic_matrix& other) const&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		dynamic_matrix temp(_rows, _columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = _data[i] + other._data[i];
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator+(const dynamic_matrix& other) &&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] += other._data[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator+(dynamic_matrix&& other) const&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			other._data[i] += _data[i];
		return std::move(other);
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator+(dynamic_matrix&& other) &&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] += other._data[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator-(const dynamic_matrix& other) const&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		dynamic_matrix temp(_rows, _columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = _data[i] - other._data[i];
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator-(const dynamic_matrix& other) &&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] -= other._data[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator-(dynamic_matrix&& other) const&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			other._data[i] = _data[i] - other._data;
		return std::move(other);
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator-(dynamic_matrix&& other)&&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] -= other._data[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator*(const dynamic_matrix& other) const&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		dynamic_matrix temp(_rows, _columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = _data[i] * other._data[i];
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator*(const dynamic_matrix& other) &&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] *= other._data[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator*(dynamic_matrix&& other) const&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			other._data[i] *= _data[i];
		return std::move(other);
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator*(dynamic_matrix&& other) &&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] *= other._data[i];
		return std::move(*this);
	}


	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator/(const dynamic_matrix& other) const&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		dynamic_matrix temp(_rows, _columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = _data[i] / other._data[i];
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator/(const dynamic_matrix& other) &&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] /= other._data[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator/(dynamic_matrix&& other) const&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			other._data[i] = _data[i] / other._data;
		return std::move(other);
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator/(dynamic_matrix&& other) &&
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] /= other._data[i];
		return std::move(*this);
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	matrix<_T, rows, columns> dynamic_matrix<_T>::operator+(const matrix<_T, rows, columns>& static_matrix) const
	{
		stm_assert(_rows == rows && _columns == columns);
		stm::matrix<_T, rows, columns> temp;
		for (unsigned int i = 0; i < static_matrix.GetSize(); ++i)
			temp[0][i] = _data[i] + static_matrix.GetData()[i];
		return temp;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	matrix<_T, rows, columns> dynamic_matrix<_T>::operator-(const matrix<_T, rows, columns>& static_matrix) const
	{
		stm_assert(_rows == rows && _columns == columns);
		stm::matrix<_T, rows, columns> temp;
		for (unsigned int i = 0; i < static_matrix.GetSize(); ++i)
			temp[0][i] = _data[i] - static_matrix.GetData()[i];
		return temp;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	matrix<_T, rows, columns> dynamic_matrix<_T>::operator*(const matrix<_T, rows, columns>& static_matrix) const
	{
		stm_assert(_rows == rows && _columns == columns);
		stm::matrix<_T, rows, columns> temp;
		for (unsigned int i = 0; i < static_matrix.GetSize(); ++i)
			temp[0][i] = _data[i] * static_matrix.GetData()[i];
		return temp;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	matrix<_T, rows, columns> dynamic_matrix<_T>::operator/(const matrix<_T, rows, columns>& static_matrix) const
	{
		stm_assert(_rows == rows && _columns == columns);
		stm::matrix<_T, rows, columns> temp;
		for (unsigned int i = 0; i < static_matrix.GetSize(); ++i)
			temp[0][i] = _data[i] / static_matrix.GetData()[i];
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator+(const _T& value) const&
	{
		dynamic_matrix temp(_rows, _columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = _data[i] + value;
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator+(const _T& value)&&
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] += value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator-(const _T& value) const&
	{
		dynamic_matrix temp(_rows, _columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = _data[i] - value;
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator-(const _T& value)&&
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] -= value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator*(const _T& value) const&
	{
		dynamic_matrix temp(_rows, _columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = _data[i] * value;
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator*(const _T& value)&&
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] *= value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::operator/(const _T& value) const&
	{
		dynamic_matrix temp(_rows, _columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			temp._data[i] = _data[i] / value;
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::operator/(const _T& value)&&
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] /= value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator+=(const dynamic_matrix& other)
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] + other._data[i];
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator-=(const dynamic_matrix& other)
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] - other._data[i];
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator*=(const dynamic_matrix& other)
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] * other._data[i];
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator/=(const dynamic_matrix& other)
	{
		stm_assert(_rows == other._rows && _columns == other._columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] / other._data[i];
		return *this;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator+=(const matrix<_T, rows, columns>& static_matrix)
	{
		stm_assert(_rows == rows && _columns == columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] + static_matrix[0][i];
		return *this;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator-=(const matrix<_T, rows, columns>& static_matrix)
	{
		stm_assert(_rows == rows && _columns == columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] - static_matrix[0][i];
		return *this;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator*=(const matrix<_T, rows, columns>& static_matrix)
	{
		stm_assert(_rows == rows && _columns == columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] * static_matrix[0][i];
		return *this;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator/=(const matrix<_T, rows, columns>& static_matrix)
	{
		stm_assert(_rows == rows && _columns == columns);
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] / static_matrix[0][i];
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator+=(const _T& value)
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] + value;
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator-=(const _T& value)
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] - value;
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator*=(const _T& value)
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] * value;
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::operator/=(const _T& value)
	{
		for (unsigned int i = 0; i < _rows * _columns; ++i)
			_data[i] = _data[i] / value;
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::Minor(unsigned int row, unsigned int column) const
	{
		stm_assert(row < _rows&& column < _columns);
		dynamic_matrix temp(_rows - 1, _columns - 1);
		unsigned int k = 0;

		for (unsigned int i = 0; i < _rows; ++i)
		{
			if (i != row)
			{
				for (unsigned int j = 0; j < _columns; ++j)
				{
					if (j != column)
					{
						temp._data[k] = _data[(i * _columns) + j];
						++k;
					}
				}
			}
		}
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::Inverse() const
	{
		dynamic_matrix temp(_rows, _columns);
		_T determinant = 0;
		for (unsigned int i = 0; i < _rows; ++i)
		{
			for (unsigned int j = 0; j < _columns; ++j)
			{
				temp._data[(i * _columns) + j] = (((i + j) % 2) ? -(this->Minor(i, j).Determinant()) : (this->Minor(i, j).Determinant()));
			}
		}

		for (unsigned int k = 0; k < _columns; ++k)
			determinant += temp._data[k] * _data[k];

		return temp.Transpose() / determinant;
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::Transpose() const&
	{
		dynamic_matrix temp(_columns, _rows);
		for (unsigned int i = 0; i < _rows; ++i)
		{
			for (unsigned int j = 0; j < _columns; ++j)
				temp._data[i + (j * _rows)] = (*this)[i][j];
		}
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>&& dynamic_matrix<_T>::Transpose() &&
	{
		for (unsigned int i = 0; i < _rows - 1; ++i)
		{
			for (unsigned int j = i + 1; j < _columns; ++j)
				std::swap(_data[i + (j * _rows)], _data[(i * _columns) + j]);
		}
		return std::move(*this);
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::SubMatrix(
		unsigned int rowSize, unsigned int columnSize, unsigned int rowOffset, unsigned int columnOffset) const
	{
		dynamic_matrix temp(rowSize, columnSize);
		for (unsigned int i = rowOffset; i < rowOffset + rowSize; ++i)
			memcpy(temp._data[i - rowOffset], _data[(i * _columns) + columnOffset], columnSize * sizeof(_T));
		return temp;
	}

	template<typename _T>
	_T dynamic_matrix<_T>::Determinant() const
	{
		stm_assert(_rows == _columns);
		if (_rows == 2)
			return (_data[0] * _data[3]) - (_data[1] * _data[2]);
		else
		{
			_T sum = 0;
			for (unsigned int i = 0; i < _rows; ++i)
				sum += (i % 2) ? -(Minor(0, i).Determinant() * _data[i]) : (Minor(0, i).Determinant() * _data[i]);
			return sum;
		}
	}

	template<typename _T>
	dynamic_matrix<_T> dynamic_matrix<_T>::Multiply(const dynamic_matrix& mat) const
	{
		stm_assert(_columns == mat._rows);
		dynamic_matrix temp(_rows, mat._columns);
		for (unsigned int i = 0; i < _rows; ++i)
		{
			for (unsigned int j = 0; j < mat._columns; ++j)
			{
				_T sum = 0;
				for (unsigned int k = 0; k < _columns; ++k)
					sum += (*this)[i][k] * mat[k][j];
				temp._data[(i * mat._columns) + j] = sum;
			}
		}
		return temp;
	}

	template<typename _T> template<unsigned int rows, unsigned int columns>
	dynamic_matrix<_T> dynamic_matrix<_T>::Multiply(const matrix<_T, rows, columns>& static_matrix) const
	{
		stm_assert(_columns == static_matrix.GetRowSize());
		dynamic_matrix temp(_rows, static_matrix.GetColumnSize());
		for (unsigned int i = 0; i < _rows; ++i)
		{
			for (unsigned int j = 0; j < static_matrix.GetColumnSize(); ++j)
			{
				_T sum = 0;
				for (unsigned int k = 0; k < _columns; ++k)
					sum += (*this)[i][k] * static_matrix[k][j];
				temp._data[(i * static_matrix.GetColumnSize()) + j] = sum;
			}
		}
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_matrix<_T>::Multiply(const dynamic_vector<_T>& vec) const
	{
		stm_assert(vec.GetSize() == _columns);

		dynamic_vector<_T> temp(_rows);
		for (unsigned int i = 0; i < _rows; ++i)
		{
			for (unsigned int j = 0; j < _columns; ++j)
				temp[i] += vec[j] * (*this)[i][j];
		}
		return temp;
	}

	template<typename _T>template<unsigned int columns>
	dynamic_vector<_T> dynamic_matrix<_T>::Multiply(const vector<_T, columns>& vec) const
	{
		stm_assert(columns == _columns);

		dynamic_vector<_T> temp(_rows);
		for (unsigned int i = 0; i < _rows; ++i)
		{
			for (unsigned int j = 0; j < _columns; ++j)
				temp[i] += vec[j] * (*this)[i][j];
		}
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_matrix<_T>::GetColumnVector(unsigned int column) const
	{
		dynamic_vector<_T> temp(_rows);
		for (unsigned int i = 0; i < _rows; ++i)
			temp[i] = (*this)[i][column];
		return temp;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetRowVector(unsigned int row, const dynamic_vector<_T>& vec)
	{
		stm_assert(vec.GetSize() == _columns);
		memcpy(&_data[row * _columns], vec.GetData(), _columns * sizeof(_T));
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetColumnVector(unsigned int column, const dynamic_vector<_T>& vec)
	{
		stm_assert(vec.GetSize() == _rows);
		for (unsigned int i = 0; i < _rows; ++i)
			(*this)[i][column] = vec[i];
		return *this;
	}

	template<typename _T> template<unsigned int columns>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetRowVector(unsigned int row, const vector<_T, columns>& vec)
	{
		stm_assert(columns == _columns);
		memcpy(&_data[row * _columns], vec.GetData(), _columns * sizeof(_T));
		return *this;
	}

	template<typename _T> template<unsigned int rows>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetColumnVector(unsigned int column, const vector<_T, rows>& vec)
	{
		stm_assert(rows == _rows);
		for (unsigned int i = 0; i < _rows; ++i)
			(*this)[i][column] = vec[i];
		return *this;
	}
	
	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetAll(_T value)
	{
		for (unsigned int i = 0; i < GetSize(); ++i)
			_data[i] = value;
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetAllRows(const dynamic_vector<_T>& vec)
	{
		stm_assert(vec.GetSize() == _columns);
		for (unsigned int i = 0; i < _rows; ++i)
			memcpy(&_data[i * _columns], vec.GetData(), _columns * sizeof(_T));
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetAllColumns(const dynamic_vector<_T>& vec)
	{
		stm_assert(vec.GetSize() == _rows);
		for (unsigned int i = 0; i < _rows; ++i)
		{
			for (unsigned int j = 0; j < _columns; ++j)
				(*this)[i][j] = vec[i];
		}
		return *this;
	}

	template<typename _T> template<unsigned int columns>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetAllRows(const vector<_T, columns>& vec)
	{
		stm_assert(vec.GetSize() == _columns);
		for (unsigned int i = 0; i < _rows; ++i)
			memcpy(&_data[i * _columns], vec.GetData(), _columns * sizeof(_T));
		return *this;
	}

	template<typename _T> template<unsigned int rows>
	dynamic_matrix<_T>& dynamic_matrix<_T>::SetAllColumns(const vector<_T, rows>& vec)
	{
		stm_assert(vec.GetSize() == _rows);
		for (unsigned int i = 0; i < _rows; ++i)
		{
			for (unsigned int j = 0; j < _columns; ++j)
				(*this)[i][j] = vec[i];
		}
		return *this;
	}

	template<typename _T>
	void dynamic_matrix<_T>::Resize(unsigned int rows, unsigned int columns)
	{
		stm_assert(rows != 0 && columns != 0);
		if (rows * columns > GetSize())
		{
			_T* newData = new _T[rows * columns];
			memset(newData, 0, sizeof(_T) * rows * columns);
			memcpy(newData, _data, rows * columns * sizeof(_T));
			delete[] _data;
			_data = newData;
		}
		_rows = rows;
		_columns = columns;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::ApplyToMatrix(_T(*func)(_T))
	{
		for (unsigned int i = 0; i < GetSize(); ++i)
			_data[i] = func(_data[i]);
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::ApplyToRow(unsigned int row, _T(*func)(_T))
	{
		for (unsigned int i = 0; i < _columns; ++i)
			_data[(row * _columns) + i] = func(_data[(row * _columns) + i]);
		return *this;
	}

	template<typename _T>
	dynamic_matrix<_T>& dynamic_matrix<_T>::ApplyToColumn(unsigned int column, _T(*func)(_T))
	{
		for (unsigned int i = 0; i < _rows; ++i)
			(*this)[i][column] = func((*this)[i][column]);
		return *this;
	}

	template<typename _T> template<typename O_TYPE>
	dynamic_matrix<O_TYPE> dynamic_matrix<_T>::Cast() const
	{
		dynamic_matrix<O_TYPE> temp(_rows, _columns);
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = O_TYPE(_data[i]);
		return temp;
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> multiply(const dynamic_matrix<_TYPE>& mat1, const dynamic_matrix<_TYPE>& mat2)
	{
		stm_assert(mat1.GetColumnSize() == mat2.GetRowSize());
		dynamic_matrix<_TYPE> temp(mat1.GetRowSize(), mat2.GetColumnSize());
		for (unsigned int i = 0; i < mat1.GetRowSize(); ++i)
		{
			for (unsigned int j = 0; j < mat2.GetColumnSize(); ++j)
			{
				_TYPE sum = 0;
				for (unsigned int k = 0; k < mat1.GetColumnSize(); ++k)
					sum += mat1[i][k] * mat2[k][j];
				temp[0][(i * mat2.GetColumnSize()) + j] = sum;
			}
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	dynamic_matrix<_TYPE> multiply(const dynamic_matrix<_TYPE>& mat1, const matrix<_TYPE, _ROWS, _COLUMNS>& mat2)
	{
		stm_assert(mat1.GetColumnSize() == mat2.GetRowSize());
		dynamic_matrix<_TYPE> temp(mat1.GetRowSize(), mat2.GetColumnSize());
		for (unsigned int i = 0; i < mat1.GetRowSize(); ++i)
		{
			for (unsigned int j = 0; j < mat2.GetColumnSize(); ++j)
			{
				_TYPE sum = 0;
				for (unsigned int k = 0; k < mat1.GetColumnSize(); ++k)
					sum += mat1[i][k] * mat2[k][j];
				temp[0][(i * mat2.GetColumnSize()) + j] = sum;
			}
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	dynamic_matrix<_TYPE> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat1, const dynamic_matrix<_TYPE>& mat2)
	{
		stm_assert(mat1.GetColumnSize() == mat2.GetRowSize());
		dynamic_matrix<_TYPE> temp(mat1.GetRowSize(), mat2.GetColumnSize());
		for (unsigned int i = 0; i < mat1.GetRowSize(); ++i)
		{
			for (unsigned int j = 0; j < mat2.GetColumnSize(); ++j)
			{
				_TYPE sum = 0;
				for (unsigned int k = 0; k < mat1.GetColumnSize(); ++k)
					sum += mat1[i][k] * mat2[k][j];
				temp[0][(i * mat2.GetColumnSize()) + j] = sum;
			}
		}
		return temp;
	}

	template<typename _TYPE>
	dynamic_vector<_TYPE> multiply(const dynamic_matrix<_TYPE>& mat, const dynamic_vector<_TYPE>& vec)
	{
		stm_assert(vec.GetSize() == mat.GetColumnSize());
		dynamic_vector<_TYPE> temp(mat.GetRowSize());
		for (unsigned int i = 0; i < mat.GetRowSize(); ++i)
		{
			_TYPE sum = 0;
			for (unsigned int j = 0; j < mat.GetColumnSize(); ++j)
				sum += mat[i][j] * vec[j];
			temp[i] = sum;
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	dynamic_vector<_TYPE> multiply(const dynamic_matrix<_TYPE>& mat, const vector<_TYPE, _DIM>& vec)
	{
		stm_assert(vec.GetSize() == mat.GetColumnSize());
		dynamic_vector<_TYPE> temp(mat.GetRowSize());
		for (unsigned int i = 0; i < mat.GetRowSize(); ++i)
		{
			_TYPE sum = 0;
			for (unsigned int j = 0; j < mat.GetColumnSize(); ++j)
				sum += mat[i][j] * vec[j];
			temp[i] = sum;
		}
		return temp;
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> pow(const dynamic_matrix<_TYPE>& mat, unsigned int power)
	{
		stm_assert(mat.GetRowSize() == mat.GetColumnSize());
		switch (power)
		{
		case 2:
			return multiply(mat, mat);
		case 3:
			return multiply(mat, multiply(mat, mat));
		default:
		{
			if (power % 2)
				return multiply(mat, multiply(pow(mat, power / 2), pow(mat, power / 2)));
			else
				return multiply(pow(mat, power / 2), pow(mat, power / 2));
		}
		}
	}

	template<typename _TYPE>
	matrix<_TYPE, 2, 2> sqrt(const dynamic_matrix<_TYPE>& mat)
	{
		stm_assert(mat.GetColumnSize() == 2 && mat.GetRowSize() == 2);
		_TYPE temp = sqrt(determinant(mat));
		return matrix<_TYPE, 2, 2>(mat[0][0] + temp, mat[0][1], mat[1][0], mat[1][1] + temp) / (sqrt(temp + 2 * (mat[0][0] + mat[1][1])));
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> transpose(const dynamic_matrix<_TYPE>& mat)
	{
		dynamic_matrix<_TYPE> temp(mat.GetColumnSize(), mat.GetRowSize());
		for (unsigned int i = 0; i < mat.GetRowSize(); ++i)
		{
			for (unsigned int j = 0; j < mat.GetColumnSize(); ++j)
				temp[0][i + (j * mat.GetRowSize())] = (*this)[i][j];
		}
		return temp;
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE>&& transpose(dynamic_matrix<_TYPE>&& mat)
	{
		for (unsigned int i = 0; i < mat.GetRowSize() - 1; ++i)
		{
			for (unsigned int j = i + 1; j < mat.GetColumnSize(); ++j)
				std::swap(mat[0][i + (j * mat.GetRowSize())], mat[0][(i * mat.GetColumnSize()) + j]);
		}
		return std::move(*this);
	}

	template<typename _TYPE>
	_TYPE determinant(const dynamic_matrix<_TYPE>& mat)
	{
		stm_assert(mat.GetRowSize() == mat.GetColumnSize());

		if (mat.GetRowSize() == 2)
			return (mat[0][0] * mat[1][1]) - (mat[0][1] * mat[1][0]);
		else
		{
			_TYPE sum = 0;
			for (unsigned int i = 0; i < mat.GetRowSize(); ++i)
				sum += (i % 2) ? -(determinant(mat.Minor(0, i)) * mat[0][i]) : (determinant(mat.Minor(0, i)) * mat[0][i]);
			return sum;
		}
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> inverse(const dynamic_matrix<_TYPE>& mat)
	{
		stm_assert(mat.GetRowSize() == mat.GetColumnSize())
			dynamic_matrix<_TYPE> temp(mat.GetRowSize(), mat.GetColumnSize());
		_TYPE det = 0;

		for (unsigned int i = 0; i < mat.GetRowSize(); ++i)
		{
			for (unsigned int j = 0; j < mat.GetColumnSize(); ++j)
				temp[0][(i * mat.GetColumnSize()) + j] = (((i + j) % 2) ? -(determinant(mat.Minor(i, j))) : (determinant(mat.Minor(i, j))));
		}

		for (unsigned int k = 0; k < mat.GetColumnSize(); ++k)
			det += temp[k] * mat[0][k];

		return transpose(temp) / det;
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> cofactorMatrix(const dynamic_matrix<_TYPE>& mat)
	{
		stm_assert(mat.GetRowSize() == mat.GetColumnSize());
		dynamic_matrix<_TYPE> temp(mat.GetRowSize(), mat.GetColumnSize());

		for (unsigned int i = 0; i < mat.GetRowSize(); ++i)
		{
			for (unsigned int j = 0; j < mat.GetColumnSize(); ++j)
			{
				temp[0][(i * mat.GetColumnSize()) + j] = (((i + j) % 2) ? -(determinant(mat.Minor(i, j))) : (determinant(mat.Minor(i, j))));
			}
		}
		return temp;
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> GetIndentityMatrix(unsigned int dimensions)
	{
		dynamic_matrix<_TYPE> mat(dimensions);
		for (unsigned int i = 0; i < dimensions; ++i)
			mat[i][i] = (_TYPE)1;
		return mat;
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> GetExchangeMatrix(unsigned int dimensions)
	{
		dynamic_matrix<_TYPE> mat(dimensions);
		for (unsigned int i = 0; i < dimensions; ++i)
			mat[i][dimensions - 1 - i] = (_TYPE)1;
		return mat;
	}
}

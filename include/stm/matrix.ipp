namespace stm
{
	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>::matrix()
	{
		memset(_data, 0, GetSize() * sizeof(_TYPE));
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>::matrix(const _TYPE& value)
	{
		std::fill_n(_data, GetSize(), value);
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>::matrix(const _TYPE data[_ROWS * _COLUMNS])
	{
		memcpy(_data, data, GetSize() * sizeof(_TYPE));
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>::matrix(const matrix& other)
	{
		memcpy(_data, other._data, GetSize() * sizeof(_TYPE));
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>::matrix(std::initializer_list<_TYPE> list)
	{
		stm_assert(list.size() == GetSize());
		std::copy(list.begin(), list.end(), _data);
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>::matrix(std::initializer_list<std::initializer_list<_TYPE>> list)
	{
		stm_assert(list.size() == GetRowSize() && list.begin()->size() == GetColumnSize());
		for (unsigned int i = 0; i < _ROWS; ++i)
			std::copy(list.begin()[i].begin(), list.begin()[i].end(), &_data[i * _COLUMNS]);
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator=(const dynamic_matrix<_TYPE>& mat)
	{
		stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
		memcpy(_data, mat.GetData(), GetSize() * sizeof(_TYPE));
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator-() const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = -_data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator+(const matrix& other) const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] + other._data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator-(const matrix& other) const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] - other._data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator*(const matrix& other) const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] * other._data[i];
		return matrix(temp);
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator/(const matrix& other) const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] / other._data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator+(const _TYPE& other) const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] + other;
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator-(const _TYPE& other) const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] - other;
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator*(const _TYPE& other) const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] * other;
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator/(const _TYPE& other) const
	{
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] / other;
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator+(const dynamic_matrix<_TYPE>& mat) const
	{
		stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] + mat.GetData()[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator-(const dynamic_matrix<_TYPE>& mat) const
	{
		stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] - mat.GetData()[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator*(const dynamic_matrix<_TYPE>& mat) const
	{
		stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] * mat.GetData()[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::operator/(const dynamic_matrix<_TYPE>& mat) const
	{
		stm_assert(mat.GetRowSize() == _ROWS && mat.GetColumnSize() == _COLUMNS);
		matrix temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = _data[i] / mat.GetData()[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator+=(const matrix& other)
	{
		*this = *this + other;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator-=(const matrix& other)
	{
		*this = *this - other;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator*=(const matrix& other)
	{
		*this = *this * other;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator/=(const matrix& other)
	{
		*this = *this / other;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator+=(const _TYPE& other)
	{
		*this = *this + other;
		return *this;
	}
	
	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator-=(const _TYPE& other)
	{
		*this = *this - other;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator*=(const _TYPE& other)
	{
		*this = *this * other;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator/=(const _TYPE& other)
	{
		*this = *this / other;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator+=(const dynamic_matrix<_TYPE>& mat)
	{
		*this = *this + mat;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator-=(const dynamic_matrix<_TYPE>& mat)
	{
		*this = *this - mat;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator*=(const dynamic_matrix<_TYPE>& mat)
	{
		*this = *this * mat;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::operator/=(const dynamic_matrix<_TYPE>& mat)
	{
		*this = *this / mat;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS - 1, _COLUMNS - 1> matrix<_TYPE, _ROWS, _COLUMNS>::Minor(unsigned int row, unsigned int column) const
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

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::Inverse() const
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

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _COLUMNS, _ROWS> matrix<_TYPE, _ROWS, _COLUMNS>::Transpose() const
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
	template<unsigned int rows, unsigned columns>
	matrix<_TYPE, rows, columns> matrix<_TYPE, _ROWS, _COLUMNS>::SubMatrix(unsigned int rowOffset, unsigned int columnOffset) const
	{
		matrix temp;
		for (unsigned int i = rowOffset; i < rowOffset + rows; ++i)
			memcpy(temp._data[i - rowOffset], _data[(i * _COLUMNS) + columnOffset], columns * sizeof(_TYPE));
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	_TYPE matrix<_TYPE, _ROWS, _COLUMNS>::Determinant() const
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");

		_TYPE sum = 0;
		for (unsigned int i = 0; i < _ROWS; ++i)
			sum += (i % 2) ? -(Minor(0, i).Determinant() * _data[i]) : (Minor(0, i).Determinant() * _data[i]);
		return sum;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS> template<unsigned int O_COLUMNS>
	matrix<_TYPE, _ROWS, O_COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::Multiply(const matrix<_TYPE, _COLUMNS, O_COLUMNS>& mat) const
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

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS> template<unsigned int O_COLUMNS>
	matrix<_TYPE, _ROWS, O_COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::mult(const matrix<_TYPE, _COLUMNS, O_COLUMNS>& mat) const
	{
		matrix<_TYPE, _ROWS, O_COLUMNS> temp;
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
			{
				for (unsigned int k = 0; k < O_COLUMNS; ++k)
					temp[i][k] += (*this)[i][j] * mat[j][k];
			}
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	vector<_TYPE, _ROWS> matrix<_TYPE, _ROWS, _COLUMNS>::Multiply(const vector<_TYPE, _COLUMNS>& vec) const
	{
		vector<_TYPE, _ROWS> temp;
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				temp[i] += vec[j] * (*this)[i][j];
		}
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	vector<_TYPE, _ROWS> matrix<_TYPE, _ROWS, _COLUMNS>::Multiply(const dynamic_vector<_TYPE>& vec) const
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

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	vector<_TYPE, _ROWS> matrix<_TYPE, _ROWS, _COLUMNS>::GetColumnVector(unsigned int column) const
	{
		vector<_TYPE, _ROWS> temp;
		for (unsigned int i = 0; i < _ROWS; ++i)
			temp[i] = (*this)[i][column];
		return temp;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::SetRowVector(unsigned int row, const vector<_TYPE, _COLUMNS>& vec)
	{
		memcpy(&_data[row * _COLUMNS], vec.GetData(), _COLUMNS * sizeof(_TYPE));
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::SetRowVector(unsigned int row, const dynamic_vector<_TYPE>& vec)
	{
		stm_assert(_COLUMNS == vec.GetSize());
		memcpy(&_data[row * _COLUMNS], vec.GetData(), _COLUMNS * sizeof(_TYPE));
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::SetColumnVector(unsigned int column, const vector<_TYPE, _ROWS>& vec)
	{
		for (unsigned int i = 0; i < _ROWS; ++i)
			(*this)[i][column] = vec[i];
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::SetColumnVector(unsigned int column, const dynamic_vector<_TYPE>& vec)
	{
		stm_assert(_ROWS == vec.GetSize());
		for (unsigned int i = 0; i < _ROWS; ++i)
			(*this)[i][column] = vec[i];
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::ApplyToMatrix(_TYPE(*func)(_TYPE))
	{
		for (unsigned int i = 0; i < GetSize(); ++i)
			_data[i] = func(_data[i]);
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::ApplyToRow(unsigned int row, _TYPE(*func)(_TYPE))
	{
		for (unsigned int i = 0; i < _COLUMNS; ++i)
			_data[(row * _COLUMNS) + i] = func(_data[(row * _COLUMNS) + i]);
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::ApplyToColumn(unsigned int column, _TYPE(*func)(_TYPE))
	{
		for (unsigned int i = 0; i < _ROWS; ++i)
			(*this)[i][column] = func((*this)[i][column]);
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS>& matrix<_TYPE, _ROWS, _COLUMNS>::SetAll(_TYPE value)
	{
		for (unsigned int i = 0; i < GetSize(); ++i)
			_data[i] = value;
		return *this;
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS> template<typename O_TYPE>
	matrix<O_TYPE, _ROWS, _COLUMNS> matrix<_TYPE, _ROWS, _COLUMNS>::Cast() const
	{
		matrix<O_TYPE, _ROWS, _COLUMNS> temp;
		for (unsigned int i = 0; i < GetSize(); ++i)
			temp._data[i] = O_TYPE(_data[i]);
		return temp;
	}

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

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _COLUMNS, _ROWS> transpose(const matrix<_TYPE, _ROWS, _COLUMNS>& mat)
	{
		matrix<_TYPE, _COLUMNS, _ROWS> temp;
		for (unsigned int i = 0; i < _ROWS; ++i)
		{
			for (unsigned int j = 0; j < _COLUMNS; ++j)
				temp[0][i + (j * _ROWS)] = mat[i][j];
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
	constexpr matrix<_TYPE, _ROWS, _COLUMNS> GetIndentityMatrix()
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		matrix<_TYPE, _ROWS, _COLUMNS> mat;
		for (unsigned int i = 0; i < _ROWS; ++i)
			mat[i][i] = (_TYPE)1;
		return mat;
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
}
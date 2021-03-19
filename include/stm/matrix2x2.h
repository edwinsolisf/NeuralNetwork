#ifndef stm_matrix_2x2_h
#define stm_matrix_2x2_h

namespace stm
{
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
			for (unsigned int i = 0; i < 2; ++i)
			{
				for (unsigned int j = 0; j < 2; ++j)
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

	template<typename _TYPE>
	inline _TYPE determinant(const matrix<_TYPE, 2, 2>& mat)
	{
		return (mat[0][0] * mat[1][1]) - (mat[0][1] * mat[1][0]);
	}

	template<typename _TYPE>
	inline matrix<_TYPE, 2, 2> sqrt(const matrix<_TYPE, 2, 2>& mat)
	{
		_TYPE temp = sqrt(determinant(mat));
		return matrix<_TYPE, 2, 2>(mat[0][0] + temp, mat[0][1], mat[1][0], mat[1][1] + temp) / (sqrt(temp + 2 * (mat[0][0] + mat[1][1])));
	}

	typedef matrix<int, 2, 2> mat2i;
	typedef matrix<float, 2, 2> mat2f;

	const mat2i identity_mat2i = GetIndentityMatrix<int, 2, 2>();
	const mat2f identity_mat2f = GetIndentityMatrix<float, 2, 2>();
}

#endif /* stm_matrix_2x2_h */
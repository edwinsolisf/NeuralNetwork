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
		matrix();
		matrix(const _TYPE& value);
		matrix(const _TYPE data[_ROWS * _COLUMNS]);
		matrix(const matrix& other);
		matrix(std::initializer_list<_TYPE> list);
		matrix(std::initializer_list<std::initializer_list<_TYPE>> list);

		//Assigment Operators
		matrix& operator=(const dynamic_matrix<_TYPE>& mat);

		//Unary Operators
        inline _TYPE* operator[](unsigned int index) { stm_assert(index < _ROWS); return &_data[index * _COLUMNS]; }
        inline const _TYPE* operator[](unsigned int index) const { stm_assert(index < _ROWS); return &_data[index * _COLUMNS]; }

		inline matrix operator+() const	{ return *this; }

		matrix operator-() const;

		//Binary Operators
		matrix operator+(const matrix& other) const;

		matrix operator-(const matrix& other) const;

		matrix operator*(const matrix& other) const;

		matrix operator/(const matrix& other) const;

		matrix operator+(const _TYPE& other) const;

		matrix operator-(const _TYPE& other) const;

		matrix operator*(const _TYPE& other) const;

		matrix operator/(const _TYPE& other) const;

		matrix operator+(const dynamic_matrix<_TYPE>& mat) const;

		matrix operator-(const dynamic_matrix<_TYPE>& mat) const;

		matrix operator*(const dynamic_matrix<_TYPE>& mat) const;

		matrix operator/(const dynamic_matrix<_TYPE>& mat) const;

		//Binary-Assigment Operators
		matrix& operator+=(const matrix& other);

		matrix& operator-=(const matrix& other);

		matrix& operator*=(const matrix& other);

		matrix& operator/=(const matrix& other);

		matrix& operator+=(const _TYPE& other);

		matrix& operator-=(const _TYPE& other);

		matrix& operator*=(const _TYPE& other);

		matrix& operator/=(const _TYPE& other);

		matrix& operator+=(const dynamic_matrix<_TYPE>& mat);

		matrix& operator-=(const dynamic_matrix<_TYPE>& mat);

		matrix& operator*=(const dynamic_matrix<_TYPE>& mat);

		matrix& operator/=(const dynamic_matrix<_TYPE>& mat);

		//Math functions
		matrix<_TYPE, _ROWS - 1, _COLUMNS - 1> Minor(unsigned int row, unsigned int column) const;

		matrix Inverse() const;

		matrix<_TYPE, _COLUMNS, _ROWS> Transpose() const;

		template<unsigned int rows, unsigned columns>
		matrix<_TYPE, rows, columns> SubMatrix(unsigned int rowOffset, unsigned int columnOffset) const;

		_TYPE Determinant() const;
        
		template<unsigned int O_COLUMNS>
		matrix<_TYPE, _ROWS, O_COLUMNS> Multiply(const matrix<_TYPE, _COLUMNS, O_COLUMNS>& mat) const;

		template<unsigned int O_COLUMNS>
		matrix<_TYPE, _ROWS, O_COLUMNS> mult(const matrix<_TYPE, _COLUMNS, O_COLUMNS>& mat) const;

		vector<_TYPE, _ROWS> Multiply(const vector<_TYPE, _COLUMNS>& vec) const;
        
		vector<_TYPE, _ROWS> Multiply(const dynamic_vector<_TYPE>& vec) const;

		//Vector Getters and Setters
		inline vector<_TYPE, _COLUMNS> GetRowVector(unsigned int row) const { return vector<_TYPE, _COLUMNS>(&_data[row * _COLUMNS]); }
		vector<_TYPE, _ROWS> GetColumnVector(unsigned int column) const;

		matrix& SetRowVector(unsigned int row, const vector<_TYPE, _COLUMNS>& vec);

		matrix& SetRowVector(unsigned int row, const dynamic_vector<_TYPE>& vec);

		matrix& SetColumnVector(unsigned int column, const vector<_TYPE, _ROWS>& vec);

		matrix& SetColumnVector(unsigned int column, const dynamic_vector<_TYPE>& vec);

		//Data manipulation functions
		matrix& ApplyToMatrix(_TYPE(*func)(_TYPE));

		matrix& ApplyToRow(unsigned int row, _TYPE(*func)(_TYPE));

		matrix& ApplyToColumn(unsigned int column, _TYPE(*func)(_TYPE));

		matrix& SetAll(_TYPE value);

		//Casting
		template<typename O_TYPE>
		matrix<O_TYPE, _ROWS, _COLUMNS> Cast() const;


		//Data Info Functions
		inline _TYPE* GetData() { return _data; }
		inline const _TYPE* GetData() const { return _data; }
		constexpr inline unsigned int GetRowSize() const { return _ROWS; }
		constexpr inline unsigned int GetColumnSize() const { return _COLUMNS; }
		constexpr inline unsigned int GetSize() const { return _ROWS * _COLUMNS; }
	};

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS, unsigned int O_COLUMNS>
	matrix<_TYPE, _ROWS, O_COLUMNS> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat1, const matrix<_TYPE, _COLUMNS, O_COLUMNS>& mat2);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	vector<_TYPE, _ROWS> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat, const vector<_TYPE, _COLUMNS>& vec);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	vector<_TYPE, _ROWS> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat, const dynamic_vector<_TYPE>& vec);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> pow(const matrix<_TYPE, _ROWS, _COLUMNS>& mat, unsigned int power);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _COLUMNS, _ROWS> transpose(const matrix<_TYPE, _ROWS, _COLUMNS>& mat);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	_TYPE determinant(const matrix<_TYPE, _ROWS, _COLUMNS>& mat);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> inverse(const matrix<_TYPE, _ROWS, _COLUMNS>& mat);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	matrix<_TYPE, _ROWS, _COLUMNS> cofactorMatrix(const matrix<_TYPE, _ROWS, _COLUMNS>& mat);

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
	constexpr matrix<_TYPE, _ROWS, _COLUMNS> GetIndentityMatrix();

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	inline constexpr matrix<_TYPE, _ROWS, _COLUMNS> GetZeroMatrix()
	{
		static_assert(_ROWS == _COLUMNS, "Error: non-square matrix");
		return matrix<_TYPE, _ROWS, _COLUMNS>();
	}

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	constexpr matrix<_TYPE, _ROWS, _COLUMNS> GetExchangeMatrix();

	typedef matrix<int, 3, 3> mat3i;
	typedef matrix<float, 3, 3> mat3f;
	typedef matrix<int, 4, 4> mat4i;
	typedef matrix<float, 4, 4> mat4f;

	const mat3i identity_mat3i = GetIndentityMatrix<int, 3, 3>();
	const mat3f identity_mat3f = GetIndentityMatrix<float, 3, 3>();
	const mat4i identity_mat4i = GetIndentityMatrix<int, 4, 4>();
	const mat4f identity_mat4f = GetIndentityMatrix<float, 4, 4>();
}

#include "matrix.ipp"
#include "matrix2x2.h"

#endif /* stm_matrix_h */
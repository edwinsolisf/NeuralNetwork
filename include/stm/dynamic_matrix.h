#ifndef dynamic_matrix_h
#define dynamic_matrix_h

#include "debug.h"
#include "matrix.h"
#include "dynamic_vector.h"

namespace stm
{
	template<typename _T>
	class dynamic_matrix
	{
	public:

		//Constructors
		dynamic_matrix(unsigned int rows, unsigned int columns);
		dynamic_matrix(unsigned int rows, unsigned int columns, const _T* data);
		dynamic_matrix(unsigned int rows, unsigned int columns, _T value);
		dynamic_matrix(unsigned int rows, unsigned int columns, std::initializer_list<_T> list);
		dynamic_matrix(const dynamic_matrix& other);
		dynamic_matrix(dynamic_matrix&& other) noexcept;
		dynamic_matrix(_T*& data, unsigned int rows, unsigned int columns);

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix(const matrix<_T, rows, columns>& static_matrix);

		//Destructor
		~dynamic_matrix();

		//Assigment Operators
		dynamic_matrix& operator=(const dynamic_matrix& other);
		dynamic_matrix& operator=(dynamic_matrix&& other) noexcept;

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator=(const matrix<_T, rows, columns>& mat);

		//Unary Operators
		inline _T* operator[](unsigned int index) { stm_assert(index < _rows); return &_data[index * _columns]; }
		inline const _T* operator[](unsigned int index) const { stm_assert(index < _rows); return &_data[index * _columns]; }

		inline dynamic_matrix operator+() const& { return *this; }
		inline dynamic_matrix&& operator+() && { return std::move(*this); }

		dynamic_matrix operator-() const&;
		dynamic_matrix&& operator-() &&;

		//Binary Operators
		dynamic_matrix operator+(const dynamic_matrix& other) const&;
		dynamic_matrix&& operator+(const dynamic_matrix& other) &&;
		dynamic_matrix&& operator+(dynamic_matrix&& other) const&;
		dynamic_matrix&& operator+(dynamic_matrix&& other) &&;

		dynamic_matrix operator-(const dynamic_matrix& other) const&;
		dynamic_matrix&& operator-(const dynamic_matrix& other) &&;
		dynamic_matrix&& operator-(dynamic_matrix&& other) const&;
		dynamic_matrix&& operator-(dynamic_matrix&& other)&&;

		dynamic_matrix operator*(const dynamic_matrix& other) const&;
		dynamic_matrix&& operator*(const dynamic_matrix& other) &&;
		dynamic_matrix&& operator*(dynamic_matrix&& other) const&;
		dynamic_matrix&& operator*(dynamic_matrix&& other)&&;

		dynamic_matrix operator/(const dynamic_matrix& other) const&;
		dynamic_matrix&& operator/(const dynamic_matrix& other) &&;
		dynamic_matrix&& operator/(dynamic_matrix&& other) const&;
		dynamic_matrix&& operator/(dynamic_matrix&& other) &&;
		
		template<unsigned int rows, unsigned int columns>
		matrix<_T, rows, columns> operator+(const matrix<_T, rows, columns>& static_matrix) const;

		template<unsigned int rows, unsigned int columns>
		matrix<_T, rows, columns> operator-(const matrix<_T, rows, columns>& static_matrix) const;

		template<unsigned int rows, unsigned int columns>
		matrix<_T, rows, columns> operator*(const matrix<_T, rows, columns>& static_matrix) const;

		template<unsigned int rows, unsigned int columns>
		matrix<_T, rows, columns> operator/(const matrix<_T, rows, columns>& static_matrix) const;


		dynamic_matrix operator+(const _T& value) const&;
		dynamic_matrix&& operator+(const _T& value) &&;

		dynamic_matrix operator-(const _T& value) const&;
		dynamic_matrix&& operator-(const _T& value) &&;

		dynamic_matrix operator*(const _T& value) const&;
		dynamic_matrix&& operator*(const _T& value) &&;

		dynamic_matrix operator/(const _T& value) const&;
		dynamic_matrix&& operator/(const _T& value) &&;


		//Binary-assigment operators
		dynamic_matrix& operator+=(const dynamic_matrix& other);

		dynamic_matrix& operator-=(const dynamic_matrix& other);

		dynamic_matrix& operator*=(const dynamic_matrix& other);

		dynamic_matrix& operator/=(const dynamic_matrix& other);
		

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator+=(const matrix<_T, rows, columns>& static_matrix);

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator-=(const matrix<_T, rows, columns>& static_matrix);

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator*=(const matrix<_T, rows, columns>& static_matrix);

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix& operator/=(const matrix<_T, rows, columns>& static_matrix);


		dynamic_matrix& operator+=(const _T& value);

		dynamic_matrix& operator-=(const _T& value);

		dynamic_matrix& operator*=(const _T& value);

		dynamic_matrix& operator/=(const _T& value);


		dynamic_matrix Minor(unsigned int row, unsigned int column) const;

		dynamic_matrix Inverse() const;
		
		dynamic_matrix Transpose() const&;
		dynamic_matrix&& Transpose() &&;

		dynamic_matrix SubMatrix(unsigned int rowSize, unsigned int columnSize, unsigned int rowOffset, unsigned int columnOffset) const;

		_T Determinant() const;

		dynamic_matrix Multiply(const dynamic_matrix& mat) const;

		template<unsigned int rows, unsigned int columns>
		dynamic_matrix Multiply(const matrix<_T, rows, columns>& static_matrix) const;

		dynamic_vector<_T> Multiply(const dynamic_vector<_T>& vec) const;

		template<unsigned int columns>
		dynamic_vector<_T> Multiply(const vector<_T, columns>& vec) const;


		//Vector Getters and Setters
		inline dynamic_vector<_T> GetRowVector(unsigned int row) const { return dynamic_vector<_T>(_columns, &_data[row * _columns]); }
		dynamic_vector<_T> GetColumnVector(unsigned int column) const;
		
		dynamic_matrix& SetRowVector(unsigned int row, const dynamic_vector<_T>& vec);

		dynamic_matrix& SetColumnVector(unsigned int column, const dynamic_vector<_T>& vec);

		template<unsigned int columns>
		dynamic_matrix& SetRowVector(unsigned int row, const vector<_T, columns>& vec);

		template<unsigned int rows>
		dynamic_matrix& SetColumnVector(unsigned int column, const vector<_T, rows>& vec);

		dynamic_matrix& SetAll(_T value);

		dynamic_matrix& SetAllRows(const dynamic_vector<_T>& vec);

		dynamic_matrix& SetAllColumns(const dynamic_vector<_T>& vec);

		template<unsigned int columns>
		dynamic_matrix& SetAllRows(const vector<_T, columns>& vec);

		template<unsigned int rows>
		dynamic_matrix& SetAllColumns(const vector<_T, rows>& vec);

		//Data manipulation functions
		void Resize(unsigned int rows, unsigned int columns);

		dynamic_matrix& ApplyToMatrix(_T(*func)(_T));

		dynamic_matrix& ApplyToRow(unsigned int row, _T(*func)(_T));

		dynamic_matrix& ApplyToColumn(unsigned int column, _T(*func)(_T));

		//Casting
		template<typename O_TYPE>
		dynamic_matrix<O_TYPE> Cast() const;

		//Data Info Functions
		inline _T* GetData() { return _data; }
		inline const _T* GetData() const { return _data; }
		inline _T*& GetRefToPtrData() { return _data; }
		inline const _T* const& GetRefToPtrData() const { return _data; }
		inline unsigned int GetRowSize() const { return _rows; }
		inline unsigned int GetColumnSize() const { return _columns; }
		inline unsigned int GetSize() const { return _rows * _columns; }

		friend dynamic_vector<_T>;
	private:
		_T* _data;
		unsigned int _rows, _columns;	
	};

	template<typename _TYPE>
	dynamic_matrix<_TYPE> multiply(const dynamic_matrix<_TYPE>& mat1, const dynamic_matrix<_TYPE>& mat2);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	dynamic_matrix<_TYPE> multiply(const dynamic_matrix<_TYPE>& mat1, const matrix<_TYPE, _ROWS, _COLUMNS>& mat2);

	template<typename _TYPE, unsigned int _ROWS, unsigned int _COLUMNS>
	dynamic_matrix<_TYPE> multiply(const matrix<_TYPE, _ROWS, _COLUMNS>& mat1, const dynamic_matrix<_TYPE>& mat2);

	template<typename _TYPE>
	dynamic_vector<_TYPE> multiply(const dynamic_matrix<_TYPE>& mat, const dynamic_vector<_TYPE>& vec);

	template<typename _TYPE, unsigned int _DIM>
	dynamic_vector<_TYPE> multiply(const dynamic_matrix<_TYPE>& mat, const vector<_TYPE, _DIM>& vec);

	template<typename _TYPE>
	dynamic_matrix<_TYPE> pow(const dynamic_matrix<_TYPE>& mat, unsigned int power);

	template<typename _TYPE>
	matrix<_TYPE, 2, 2> sqrt(const dynamic_matrix<_TYPE>& mat);

	template<typename _TYPE>
	dynamic_matrix<_TYPE> transpose(const dynamic_matrix<_TYPE>& mat);

	template<typename _TYPE>
	dynamic_matrix<_TYPE>&& transpose(dynamic_matrix<_TYPE>&& mat);

	template<typename _TYPE>
	_TYPE determinant(const dynamic_matrix<_TYPE>& mat);

	template<typename _TYPE>
	dynamic_matrix<_TYPE> inverse(const dynamic_matrix<_TYPE>& mat);

	template<typename _TYPE>
	dynamic_matrix<_TYPE> cofactorMatrix(const dynamic_matrix<_TYPE>& mat);

	template<typename _TYPE>
	inline dynamic_matrix<_TYPE> adjugate(const dynamic_matrix<_TYPE>& mat) {
		return transpose(cofactorMatrix(mat));
	}

	template<typename _TYPE>
	inline dynamic_matrix<_TYPE> adjugate(dynamic_matrix<_TYPE>&& mat) {
		return transpose(cofactorMatrix(mat));
	}

	template<typename _TYPE>
	inline dynamic_matrix<_TYPE> toRowMatrix(const dynamic_vector<_TYPE>& vec) {
		return dynamic_matrix<_TYPE>(1, vec.GetSize(), vec.GetData());
	}

	template<typename _TYPE>
	inline dynamic_matrix<_TYPE> toRowMatrix(dynamic_vector<_TYPE>&& vec) {
		return dynamic_matrix<_TYPE>(vec.GetRefToPtrData(), 1, vec.GetSize());
	}

	template<typename _TYPE>
	inline dynamic_matrix<_TYPE> toColumnMatrix(const dynamic_vector<_TYPE>& vec) {
		return dynamic_matrix<_TYPE>(vec.GetSize(), 1, vec.GetData());
	}

	template<typename _TYPE>
	inline dynamic_matrix<_TYPE> toColumnMatrix(dynamic_vector<_TYPE>&& vec) {
		return dynamic_matrix<_TYPE>(vec.GetRefToPtrData(), vec.GetSize(), 1);
	}

	template<typename _TYPE>
	inline dynamic_vector<_TYPE> toRowVector(const dynamic_matrix<_TYPE>& mat) {
		return dynamic_vector<_TYPE>(mat.GetSize(), mat.GetData());
	}

	template<typename _TYPE>
	inline dynamic_vector<_TYPE> toRowVector(dynamic_matrix<_TYPE>&& mat) {
		return dynamic_vector<_TYPE>(mat.GetRefToPtrData(), mat.GetSize());
	}

	template<typename _TYPE>
	inline dynamic_vector<_TYPE> toColumnVector(const dynamic_matrix<_TYPE>& mat) {
		return dynamic_vector<_TYPE>(mat.GetSize(), mat.Transpose().GetData());
	}

	template<typename _TYPE>
	inline dynamic_vector<_TYPE> toColumnVector(dynamic_matrix<_TYPE>&& mat) {
		return dynamic_vector<_TYPE>(mat.Transpose().GetRefToPtrData(), mat.GetSize());
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> GetIndentityMatrix(unsigned int dimensions);

	template<typename _TYPE>
	inline dynamic_matrix<_TYPE> GetZeroMatrix(unsigned int dimensions) {
		return dynamic_matrix<_TYPE>(dimensions);
	}

	template<typename _TYPE>
	dynamic_matrix<_TYPE> GetExchangeMatrix(unsigned int dimensions);

	typedef dynamic_matrix<int> mat_i;
	typedef dynamic_matrix<float> mat_f;

}

#include "dynamic_matrix.ipp"

#endif /* stm_dynamic_matrix_h */
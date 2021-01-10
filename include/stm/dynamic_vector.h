#ifndef stm_dynamic_vector_h
#define stm_dynamic_vector_h

#include "vector.h"

namespace stm
{
	template<typename _T>
	class dynamic_vector
	{
	public:

		//Constructors
		dynamic_vector(unsigned int dimensions);
		dynamic_vector(unsigned int dimensions, const _T* data);
		dynamic_vector(unsigned int dimensions, _T value);
		dynamic_vector(unsigned int dimensions, std::initializer_list<_T> list);
		dynamic_vector(const dynamic_vector& other);
		dynamic_vector(dynamic_vector&& other) noexcept;
		dynamic_vector(_T*& data, unsigned int dimensions);
		template<unsigned int dimensions>
		dynamic_vector(const vector<_T, dimensions>& static_vector);

		//Destructors
		~dynamic_vector();

		//Assignment operators
		dynamic_vector& operator=(const dynamic_vector& other);
		dynamic_vector& operator=(dynamic_vector&& other) noexcept;

		//Unary operators
		inline dynamic_vector operator+() const& { return *this; }
		inline dynamic_vector&& operator+()&& { return std::move(*this); }

		dynamic_vector operator-() const&;
		dynamic_vector&& operator-()&&;

		inline _T& operator[](unsigned int index) { stm_assert(index < _dimensions); return _data[index]; }
		inline const _T& operator[](unsigned int index) const { stm_assert(index < _dimensions); return _data[index]; }

		//Data Manipulation Functions
		void Resize(unsigned int dimensions);

		dynamic_vector& SetAll(_T value)&;
		dynamic_vector&& SetAll(_T value)&&;
		dynamic_vector& ApplyToVector(_T(*func)(_T))&;
		dynamic_vector&& ApplyToVector(_T(*func)(_T))&&;

		//Binary Operators
		dynamic_vector operator+(const dynamic_vector& other) const&;
		dynamic_vector&& operator+(const dynamic_vector& other)&&;
		dynamic_vector&& operator+(dynamic_vector&& other) const&;
		dynamic_vector&& operator+(dynamic_vector&& other)&&;

		dynamic_vector operator-(const dynamic_vector& other) const&;
		dynamic_vector&& operator-(const dynamic_vector& other)&&;
		dynamic_vector&& operator-(dynamic_vector&& other) const&;
		dynamic_vector&& operator-(dynamic_vector&& other)&&;

		dynamic_vector operator*(const dynamic_vector& other) const&;
		dynamic_vector&& operator*(const dynamic_vector& other)&&;
		dynamic_vector&& operator*(dynamic_vector&& other) const&;
		dynamic_vector&& operator*(dynamic_vector&& other)&&;

		dynamic_vector operator/(const dynamic_vector& other) const&;
		dynamic_vector&& operator/(const dynamic_vector& other)&&;
		dynamic_vector&& operator/(dynamic_vector&& other) const&;
		dynamic_vector&& operator/(dynamic_vector&& other)&&;

		template<unsigned int dimensions>
		vector<_T, dimensions> operator+(const vector<_T, dimensions>& static_vector) const;

		template<unsigned int dimensions>
		vector<_T, dimensions> operator-(const vector<_T, dimensions>& static_vector) const;

		template<unsigned int dimensions>
		vector<_T, dimensions> operator*(const vector<_T, dimensions>& static_vector) const;

		template<unsigned int dimensions>
		vector<_T, dimensions> operator/(const vector<_T, dimensions>& static_vector) const;

		dynamic_vector operator+(const _T& value) const&;
		dynamic_vector&& operator+(const _T& value)&&;

		dynamic_vector operator-(const _T& value) const&;
		dynamic_vector&& operator-(const _T& value)&&;

		dynamic_vector operator*(const _T& value) const&;
		dynamic_vector&& operator*(const _T& value)&&;

		dynamic_vector operator/(const _T& value) const&;
		dynamic_vector&& operator/(const _T& value)&&;


		//Binary Assigment Operators
		dynamic_vector& operator+=(const dynamic_vector& other);

		dynamic_vector& operator-=(const dynamic_vector& other);

		dynamic_vector& operator*=(const dynamic_vector& other);

		dynamic_vector& operator/=(const dynamic_vector& other);

		template<unsigned int dimensions>
		dynamic_vector& operator+=(const vector<_T, dimensions>& static_vector);

		template<unsigned int dimensions>
		dynamic_vector& operator-=(const vector<_T, dimensions>& static_vector);

		template<unsigned int dimensions>
		dynamic_vector& operator*=(const vector<_T, dimensions>& static_vector);

		template<unsigned int dimensions>
		dynamic_vector& operator/=(const vector<_T, dimensions>& static_vector);

		dynamic_vector& operator+=(const _T& value);

		dynamic_vector& operator-=(const _T& value);

		dynamic_vector& operator*=(const _T& value);

		dynamic_vector& operator/=(const _T& value);

		//Casting
		template<typename O_TYPE>
		dynamic_vector<O_TYPE> Cast() const;

		//Data Info Functions
		inline _T* GetData() { return _data; }
		inline const _T* GetData() const { return _data; }
		inline _T*& GetRefToPtrData() { return _data; }
		inline const _T* const& GetRefToPtrData() const { return _data; }
		inline unsigned int GetSize() const { return _dimensions; }

		//Math Functions
		inline _T Magnitude() const { return sqrt(DotProduct(*this)); }

		inline dynamic_vector UnitVector() const& { return (*this) / Magnitude(); }
		inline dynamic_vector&& UnitVector() && { return std::move((*this) / Magnitude()); }

		_T DotProduct(const dynamic_vector& other) const;

		template<unsigned int dimensions>
		_T DotProduct(const vector<_T, dimensions>& other) const;

		inline vector<_T, 3> CrossProduct(const dynamic_vector<_T>& vec) { stm_assert(_dimensions == 3 && vec.GetSize() == 3);
			return vector<_T, 3>((_data[1] * vec[2]) - (_data[2] * vec[1]),
								 (_data[2] * vec[0]) - (_data[0] * vec[2]),
								 (_data[0] * vec[1]) - (_data[1] * vec[0]));
		}

		inline vector<_T, 3> CrossProduct(const vector<_T, 3>& vec) { stm_assert(_dimensions == 3 && vec.GetSize() == 3);
			return vector<_T, 3>((_data[1] * vec[2]) - (_data[2] * vec[1]),
							     (_data[2] * vec[0]) - (_data[0] * vec[2]),
								 (_data[0] * vec[1]) - (_data[1] * vec[0]));
		}

	private:
		_T* _data;
		unsigned int _dimensions;
	};

	template<typename _TYPE>
	_TYPE dotproduct(const dynamic_vector<_TYPE>& vec1, const dynamic_vector<_TYPE>& vec2);
	template<typename _TYPE, unsigned int _DIM>
	_TYPE dotproduct(const vector<_TYPE, _DIM>& vec1, const dynamic_vector<_TYPE>& vec2);
	template<typename _TYPE, unsigned int _DIM>
	_TYPE dotproduct(const dynamic_vector<_TYPE>& vec1, const vector<_TYPE, _DIM>& vec2);

	template<typename _TYPE>
	inline dynamic_vector<_TYPE> pow(const dynamic_vector<_TYPE>& vec, unsigned int power) {
		dynamic_vector<_TYPE> out = vec;
		for (unsigned int i = 0; i < power; ++i)
			out *= vec;
		return out;
	}

	template<typename _TYPE>
	inline vector<_TYPE, 3> crossproduct(const dynamic_vector<_TYPE>& vec1, const dynamic_vector<_TYPE>& vec2) {
		stm_assert(vec1.GetSize() == 3 && vec2.GetSize() == 3);
		return vector<_TYPE, 3>((vec1[1] * vec2[2]) - (vec1[2] * vec2[1]),
								(vec1[2] * vec2[0]) - (vec1[0] * vec2[2]),
								(vec1[0] * vec2[1]) - (vec1[1] * vec2[0]));
	}

	template<typename _TYPE>
	inline vector<_TYPE, 3> crossproduct(const vector<_TYPE, 3>& vec1, const dynamic_vector<_TYPE>& vec2) {
		stm_assert(vec1.GetSize() == 3 && vec2.GetSize() == 3);
		return vector<_TYPE, 3>((vec1[1] * vec2[2]) - (vec1[2] * vec2[1]),
								(vec1[2] * vec2[0]) - (vec1[0] * vec2[2]),
								(vec1[0] * vec2[1]) - (vec1[1] * vec2[0]));
	}

	template<typename _TYPE>
	inline vector<_TYPE, 3> crossproduct(const dynamic_vector<_TYPE>& vec1, const vector<_TYPE, 3>& vec2) {
		stm_assert(vec1.GetSize() == 3 && vec2.GetSize() == 3);
		return vector<_TYPE, 3>((vec1[1] * vec2[2]) - (vec1[2] * vec2[1]),
								(vec1[2] * vec2[0]) - (vec1[0] * vec2[2]),
								(vec1[0] * vec2[1]) - (vec1[1] * vec2[0]));
	}

	template<typename _TYPE>
	inline _TYPE magnitude(const dynamic_vector<_TYPE>& vec) { return sqrt(dotproduct(vec, vec)); }

	template<typename _TYPE>
	inline dynamic_vector<_TYPE> normalize(const dynamic_vector<_TYPE>& vec) { return vec / magnitude(vec); }
	template<typename _TYPE>
	inline dynamic_vector<_TYPE> normalize(dynamic_vector<_TYPE>&& vec) { return std::move(vec / magnitude(vec)); }
	typedef dynamic_vector<float> vec_f;
	typedef dynamic_vector<int> vec_i;

}

#include "dynamic_vector.ipp"

#endif /* stm_dynamic_vector_h */

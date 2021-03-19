#ifndef stm_vector_h
#define stm_vector_h

#include <iostream>
#include "debug.h"

namespace stm
{
	template<typename _TYPE>
	class dynamic_vector;

	template<typename _TYPE, unsigned int _DIM>
	class vector
	{
		public:

		//Constructors
		vector();
		vector(const _TYPE data[_DIM]);
		vector(const _TYPE& value);
		vector(const vector& other);
		vector(std::initializer_list<_TYPE> list);

		vector& operator=(const dynamic_vector<_TYPE>& vec);

		//Unary Operators
		inline vector operator+() const { return *this; }

		vector operator-() const;

        inline _TYPE& operator[](const unsigned int& index) { stm_assert(index < _DIM); return _data[index]; }
        inline const _TYPE& operator[](const unsigned int& index) const { stm_assert(index < _DIM); return _data[index]; }

		//Casting
		template<unsigned int DIM>
		inline vector<_TYPE, DIM> ToVector() const {
			static_assert(DIM <= _DIM, "New vector is of greater dimensions");
			return DIM == _DIM ? *this : vector<_TYPE, DIM>(_data);
		}

		template<typename O_TYPE>
		vector<O_TYPE, _DIM> Cast() const;

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(_TYPE));

		vector& SetAll(_TYPE value);

		//Binary Operators
		vector operator+(const vector& other) const;

		vector operator-(const vector& other) const;

		vector operator*(const vector& other) const;

		vector operator/(const vector& other) const;

		vector operator+(const _TYPE& other) const;

		vector operator-(const _TYPE& other) const;

		vector operator*(const _TYPE& other) const;

		vector operator/(const _TYPE& other) const;

		vector operator+(const dynamic_vector<_TYPE>& vec) const;

		vector operator-(const dynamic_vector<_TYPE>& vec) const;

		vector operator*(const dynamic_vector<_TYPE>& vec) const;

		vector operator/(const dynamic_vector<_TYPE>& vec) const;

		//Binary assigment operators
		vector& operator+=(const vector& other);

		vector& operator-=(const vector& other);

		vector& operator*=(const vector& other);

		vector& operator/=(const vector& other);

		vector& operator+=(const _TYPE& other);

		vector& operator-=(const _TYPE& other);

		vector& operator*=(const _TYPE& other);

		vector& operator/=(const _TYPE& other);

		vector& operator+=(const dynamic_vector<_TYPE>& vec);

		vector& operator-=(const dynamic_vector<_TYPE>& vec);

		vector& operator*=(const dynamic_vector<_TYPE>& vec);

		vector& operator/=(const dynamic_vector<_TYPE>& vec);

		//Data Info Functions
		inline _TYPE* GetData() { return _data; }
		inline const _TYPE* GetData() const { return _data; }
		constexpr unsigned int GetSize() const { return _DIM; }

		//Math functions
		inline _TYPE Magnitude() const {
			return sqrt(DotProduct(*this));
		}

		inline vector UnitVector() const {
			return (*this) / Magnitude();
		}

		_TYPE DotProduct(const vector& other) const;
		_TYPE DotProduct(const dynamic_vector<_TYPE>& vec) const;

	private:
		_TYPE _data[_DIM];

	};

	template<typename _TYPE, unsigned int _DIM>
	_TYPE dotproduct(const vector<_TYPE, _DIM>& vec1, const vector<_TYPE, _DIM>& vec2);

	template<typename _TYPE>
	inline vector<_TYPE, 3> crossproduct(const vector<_TYPE, 3>& vec1, const vector<_TYPE, 3>& vec2)
	{
		return vector<_TYPE, 3>((vec1.y * vec2.z) - (vec1.z * vec2.y),
								(vec1.z * vec2.x) - (vec1.x * vec2.z),
								(vec1.x * vec2.y) - (vec1.y * vec2.x));
	}

    template<typename _TYPE, unsigned int _DIM>
    inline _TYPE magnitude(const vector<_TYPE, _DIM>& vec) {
        return sqrt(dotproduct(vec, vec));
    }

    template<typename _TYPE, unsigned int _DIM>
    inline vector<_TYPE, _DIM> normalize(const vector<_TYPE, _DIM>& vec) {
        return vec / magnitude(vec);
    }

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> pow(const vector<_TYPE, _DIM>& vec, unsigned int power);
}

#include "vector.ipp"
#include "vector4.h"
#include "vector3.h"
#include "vector2.h"

#endif /* stm_vector_h */
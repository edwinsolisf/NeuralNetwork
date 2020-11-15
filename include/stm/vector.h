#ifndef stm_vector_h
#define stm_vector_h

#include <iostream>
#include "debug.h"

namespace stm
{
	template<typename _TYPE, unsigned int _DIM>
	class vector
	{
	private:
		_TYPE _data[_DIM];

	public:

		//Constructors
		vector()
		{
			memset(_data, 0, _DIM * sizeof(_TYPE));
		}

		vector(const _TYPE data[_DIM])
		{
			memcpy(_data, data, _DIM * sizeof(_TYPE));
		}

		vector(const _TYPE& value)
		{
			for (unsigned int i = 0; i < _DIM; ++i)
				_data[i] = value;
		}

		vector(const vector& other)
		{
			memcpy(_data, other._data, _DIM * sizeof(_TYPE));
		}

		//Unary Operators
		inline vector operator+() const
		{
			return *this;
		}

		vector operator-() const
		{
			_TYPE data[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				data[i] = -_data[i];
			return vector(data);
		}

        inline _TYPE& operator[](const unsigned int& index) { stm_assert(index < _DIM); return _data[index]; }
        inline const _TYPE& operator[](const unsigned int& index) const { stm_assert(index < _DIM); return _data[index]; }

		//Casting
		template<unsigned int DIM>
		inline vector<_TYPE, DIM> ToVector() const
		{
			static_assert(DIM <= _DIM, "New vector is of greater dimensions");
			return DIM == _DIM ? *this : vector<_TYPE, DIM>(_data);
		}

		template<typename O_TYPE>
		vector<O_TYPE, _DIM> Cast() const
		{
			_TYPE data[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				data[i] = O_TYPE(_data[i]);
			return vector<O_TYPE, _DIM>(data);
		}

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < _DIM; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		//Binary Operators
		vector operator+(const vector& other) const
		{
			_TYPE temp[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = _data[i] + other._data[i];
			return vector(temp);
		}


		vector operator-(const vector& other) const
		{
			_TYPE temp[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = _data[i] - other._data[i];
			return vector(temp);
		}

		vector operator*(const vector& other) const
		{
			_TYPE temp[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = _data[i] * other._data[i];
			return vector(temp);
		}

		vector operator/(const vector& other) const
		{
			_TYPE temp[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = _data[i] / other._data[i];
			return vector(temp);
		}

		vector operator+(const _TYPE& other) const
		{
			_TYPE temp[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = _data[i] + other;
			return vector(temp);
		}

		vector operator-(const _TYPE& other) const
		{
			_TYPE temp[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = _data[i] - other;
			return vector(temp);
		}

		vector operator*(const _TYPE& other) const
		{
			_TYPE temp[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = _data[i] * other;
			return vector(temp);
		}

		vector operator/(const _TYPE& other) const
		{
			_TYPE temp[_DIM];
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = _data[i] / other;
			return vector(temp);
		}

		//Binary assigment operators
		vector& operator+=(const vector& other)
		{
			*this = *this + other;
			return *this;
		}

		vector& operator-=(const vector& other)
		{
			*this = *this - other;
			return *this;
		}

		vector& operator*=(const vector& other)
		{
			*this = *this * other;
			return *this;
		}

		vector& operator/=(const vector& other)
		{
			*this = *this / other;
			return *this;
		}

		vector& operator+=(const _TYPE& other)
		{
			*this = *this + other;
			return *this;
		}

		vector& operator-=(const _TYPE& other)
		{
			*this = *this - other;
			return *this;
		}

		vector& operator*=(const _TYPE& other)
		{
			*this = *this * other;
			return *this;
		}

		vector& operator/=(const _TYPE& other)
		{
			*this = *this / other;
			return *this;
		}

		//Data Info Functions
		inline _TYPE* GetData() { return _data; }
		inline const _TYPE* GetData() const { return _data; }
		constexpr unsigned int GetSize() const { return _DIM; }

		//Math functions
		inline _TYPE Magnitude() const
		{
			return sqrt(DotProduct(*this));
		}

		inline vector UnitVector() const
		{
			return (*this) / Magnitude();
		}

		_TYPE DotProduct(const vector& other) const
		{
			_TYPE sum = 0;
			for (unsigned int i = 0; i < _DIM; ++i)
				sum += _data[i] * other[i];
			return sum;
		}

	};

	template<typename _TYPE>
	class vector<_TYPE, 4>
	{
	public:
		struct
		{
			union
			{
				_TYPE _data[4];
				struct 
				{
					_TYPE x, y, z, w;
				};
			};
		};

		//Constructors
		vector()
		{
			memset(_data, 0, 4 * sizeof(_TYPE));
		}

		vector(const _TYPE data[4])
		{
			memcpy(_data, data, 4 * sizeof(_TYPE));
		}

		vector(const _TYPE& value)
			:w(value), x(value), y(value), z(value)
		{
		}

		vector(const _TYPE& _x, const _TYPE& _y, const _TYPE& _z, const _TYPE& _w)
			:x(_x), y(_y), z(_z), w(_w)
		{
		}

		vector(const vector& other)
		{
			memcpy(_data, other._data, 4 * sizeof(_TYPE));
		}

		//Unary Operators
		inline vector operator+() const
		{
			return *this;
		}

		inline vector operator-() const
		{
			return vector(-x, -y, -z, -w);
		}

        inline _TYPE& operator[](const unsigned int& index) { stm_assert(index < 4); return _data[index]; }
        inline const _TYPE& operator[](const unsigned int& index) const { stm_assert(index < 4); return _data[index]; }

		//Casting
		template<unsigned int DIM>
		inline vector<_TYPE, DIM> ToVector() const
		{
			static_assert(DIM <= 4, "New vector is of greater dimensions");
			return vector<_TYPE, DIM>(_data);
		}

		template<typename O_TYPE>
		vector<O_TYPE, 4> Cast() const
		{
			_TYPE data[4];
			for (unsigned int i = 0; i < 4; ++i)
				data[i] = O_TYPE(_data[i]);
			return vector<O_TYPE, 4>(data);
		}

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < 4; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		//Binary Operators
		inline vector operator+(const vector& other) const
		{
			return vector(x + other.x, y + other.y, z + other.z, w + other.w);
		}

		inline vector operator-(const vector& other) const
		{
			return vector(x - other.x, y - other.y, z - other.z, w - other.w);
		}

		inline vector operator*(const vector& other) const
		{
			return vector(x * other.x, y * other.y, z * other.z, w * other.w);
		}

		inline vector operator/(const vector& other) const
		{
			return vector(x / other.x, y / other.y, z / other.z, w / other.w);
		}

		inline vector operator+(const _TYPE& other) const
		{
			return vector(x + other, y + other, z + other, w + other);
		}

		inline vector operator-(const _TYPE& other) const
		{
			return vector(x - other, y - other, z - other, w - other);
		}

		inline vector operator*(const _TYPE& other) const
		{
			return vector(x * other, y * other, z * other, w * other);
		}

		inline vector operator/(const _TYPE& other) const
		{
			return vector(x / other, y / other, z / other, w / other);
		}

		//Binary assigment operators
		vector& operator+=(const vector& other)
		{
			*this = *this + other;
			return *this;
		}

		vector& operator-=(const vector& other)
		{
			*this = *this - other;
			return *this;
		}

		vector& operator*=(const vector& other)
		{
			*this = *this * other;
			return *this;
		}

		vector& operator/=(const vector& other)
		{
			*this = *this / other;
			return *this;
		}

		vector& operator+=(const _TYPE& other)
		{
			*this = *this + other;
			return *this;
		}

		vector& operator-=(const _TYPE& other)
		{
			*this = *this - other;
			return *this;
		}

		vector& operator*=(const _TYPE& other)
		{
			*this = *this * other;
			return *this;
		}

		vector& operator/=(const _TYPE& other)
		{
			*this = *this / other;
			return *this;
		}

		//Data Info Functions
		inline _TYPE* GetData() { return _data; }
		inline const _TYPE* GetData() const { return _data; }
		constexpr unsigned int GetSize() const { return 4; }

		//Math functions
		inline _TYPE Magnitude() const
		{
			return sqrt(DotProduct(*this));
		}

		inline vector UnitVector() const
		{
			return (*this) / Magnitude();
		}
		
		inline _TYPE DotProduct(const vector& other) const
		{
			return (x * other.x) + (y * other.y) + (z * other.z) + (w * other.w);
		}
	};

	template<typename _TYPE>
	class vector<_TYPE, 3>
	{
	public:

		struct
		{
			union
			{
				_TYPE _data[3];
				struct
				{
					_TYPE x, y, z;
				};
			};
		};

		//Constructors
		vector()
		{
			memset(_data, 0, 3 * sizeof(_TYPE));
		}

		vector(const _TYPE data[3])
		{
			memcpy(_data, data, 3 * sizeof(_TYPE));
		}

		vector(const _TYPE& _x, const _TYPE& _y, const _TYPE& _z)
			:x(_x), y(_y), z(_z)
		{
		}

		vector(const _TYPE& value)
			:x(value), y(value), z(value)
		{
		}

		vector(const vector& other)
		{
			memcpy(_data, other._data, 3 * sizeof(_TYPE));
		}

		//Unary Operators
		inline vector operator+() const
		{
			return *this;
		}

		inline vector operator-() const
		{
			return vector(-x, -y, -z);
		}

        inline _TYPE& operator[](const unsigned int& index) { stm_assert(index < 3); return _data[index]; }
        inline const _TYPE& operator[](const unsigned int& index) const { stm_assert(index < 3); return _data[index]; }

		//Casting
		template<unsigned int DIM>
		inline vector<_TYPE, DIM> ToVector() const
		{
			static_assert(DIM <= 3, "New vector is of greater dimensions");
			return DIM == 3 ? *this : vector<_TYPE, 3>(_data);
		}

		template<typename O_TYPE>
		vector<O_TYPE, 3> Cast() const
		{
			_TYPE data[3];
			for (unsigned int i = 0; i < 3; ++i)
				data[i] = O_TYPE(_data[i]);
			return vector<O_TYPE, 3>(data);
		}

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < 3; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		//Binary Operators
		inline vector operator+(const vector& other) const
		{
			return vector(x + other.x, y + other.y, z + other.z);
		}

		inline vector operator-(const vector& other) const
		{
			return vector(x - other.x, y - other.y, z - other.z);
		}

		inline vector operator*(const vector& other) const
		{
			return vector(x * other.x, y * other.y, z * other.z);
		}

		inline vector operator/(const vector& other) const
		{
			return vector(x / other.x, y / other.y, z / other.z);
		}

		inline vector operator+(const _TYPE& other) const
		{
			return vector(x + other, y + other, z + other);
		}

		inline vector operator-(const _TYPE& other) const
		{
			return vector(x - other, y - other, z - other);
		}

		inline vector operator*(const _TYPE& other) const
		{
			return vector(x * other, y * other, z * other);
		}

		inline vector operator/(const _TYPE& other) const
		{
			return vector(x / other, y / other, z / other);
		}

		//Binary assigment operators
		vector& operator+=(const vector& other)
		{
			*this = *this + other;
			return *this;
		}

		vector& operator-=(const vector& other)
		{
			*this = *this - other;
			return *this;
		}

		vector& operator*=(const vector& other)
		{
			*this = *this * other;
			return *this;
		}

		vector& operator/=(const vector& other)
		{
			*this = *this / other;
			return *this;
		}

		vector& operator+=(const _TYPE& other)
		{
			*this = *this + other;
			return *this;
		}

		vector& operator-=(const _TYPE& other)
		{
			*this = *this - other;
			return *this;
		}

		vector& operator*=(const _TYPE& other)
		{
			*this = *this * other;
			return *this;
		}

		vector& operator/=(const _TYPE& other)
		{
			*this = *this / other;
			return *this;
		}

		//Data Info Functions
		inline _TYPE* GetData() { return _data; }
		inline const _TYPE* GetData() const { return _data; }
		constexpr unsigned int GetSize() const { return 3; }

		//Math functions
		inline _TYPE Magnitude() const
		{
			return sqrt(DotProduct(*this));
		}

		inline vector UnitVector() const
		{
			return (*this) / Magnitude();
		}

		inline _TYPE DotProduct(const vector& other) const
		{
			return (x * other.x) + (y * other.y) + (z * other.z);
		}

		inline vector CrossProduct(const vector& other) const
		{
			return vector((y * other.z) - (z * other.y),
						  (z * other.x) - (x * other.z),
						  (x * other.y) - (y * other.x));
		}
	};

	template<typename _TYPE>
	class vector<_TYPE, 2>
	{
	public:
		struct
		{
			union
			{
				_TYPE _data[2];
				struct
				{
					_TYPE x, y;
				};
			};
		};

		//Constructors
		vector()
		{
			memset(_data, 0, 2 * sizeof(_TYPE));
		}

		vector(const _TYPE data[2])
		{
			memcpy(_data, data, 2 * sizeof(_TYPE));
		}

		vector(const _TYPE& _x, const _TYPE& _y)
			:x(_x), y(_y)
		{
		}

		vector(const _TYPE& value)
			:x(value), y(value)
		{
		}

		vector(const vector& other)
		{
			memcpy(_data, other._data, 2 * sizeof(_TYPE));
		}

		//Unary Operators
		inline vector operator+() const
		{
			return *this;
		}

		inline vector operator-() const
		{
			return vector(-x, -y);
		}

        inline _TYPE& operator[](const unsigned int& index) { stm_assert(index < 2); return _data[index]; }
        inline const _TYPE& operator[](const unsigned int& index) const { stm_assert(index < 2); return _data[index]; }

		template<typename O_TYPE>
		vector<O_TYPE, 2> Cast() const
		{
			_TYPE data[2];
			for (unsigned int i = 0; i < 2; ++i)
				data[i] = O_TYPE(_data[i]);
			return vector<O_TYPE, 2>(data);
		}

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(const _TYPE&))
		{
			for (unsigned int i = 0; i < 2; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		//Binary Operators
		inline vector operator+(const vector& other) const
		{
			return vector(x + other.x, y + other.y);
		}

		inline vector operator-(const vector& other) const
		{
			return vector(x - other.x, y - other.y);
		}

		inline vector operator*(const vector& other) const
		{
			return vector(x * other.x, y * other.y);
		}

		inline vector operator/(const vector& other) const
		{
			return vector(x / other.x, y / other.y);
		}

		inline vector operator+(const _TYPE& other) const
		{
			return vector(x + other, y + other);
		}

		inline vector operator-(const _TYPE& other) const
		{
			return vector(x - other, y - other);
		}

		inline vector operator*(const _TYPE& other) const
		{
			return vector(x * other, y * other);
		}

		inline vector operator/(const _TYPE& other) const
		{
			return vector(x / other, y / other);
		}

		//Binary assigment operators
		vector& operator+=(const vector& other)
		{
			*this = *this + other;
			return *this;
		}

		vector& operator-=(const vector& other)
		{
			*this = *this - other;
			return *this;
		}

		vector& operator*=(const vector& other)
		{
			*this = *this * other;
			return *this;
		}

		vector& operator/=(const vector& other)
		{
			*this = *this / other;
			return *this;
		}

		vector& operator+=(const _TYPE& other)
		{
			*this = *this + other;
			return *this;
		}

		vector& operator-=(const _TYPE& other)
		{
			*this = *this - other;
			return *this;
		}

		vector& operator*=(const _TYPE& other)
		{
			*this = *this * other;
			return *this;
		}

		vector& operator/=(const _TYPE& other)
		{
			*this = *this / other;
			return *this;
		}

		//Data Info Functions
		inline _TYPE* GetData() { return _data; }
		inline const _TYPE* GetData() const { return _data; }
		constexpr unsigned int GetSize() const { return 2; }

		//Math functions
		inline _TYPE Magnitude() const
		{
			return sqrt(DotProduct(*this));
		}

		inline vector UnitVector() const
		{
			return (*this) / Magnitude();
		}
		
		inline _TYPE DotProduct(const vector& other) const
		{
			return (x * other.x) + (y * other.y);
		}

		inline vector<_TYPE, 3> CrossProduct(const vector& other) const
		{
			return vector(0, 0, (x * other.y) - (y * other.x));
		}
	};

	template<typename _TYPE, unsigned int _DIM>
	_TYPE dotproduct(const vector<_TYPE, _DIM>& vec1, const vector<_TYPE, _DIM>& vec2)
	{
		_TYPE sum = 0;
		for (unsigned int i = 0; i < _DIM; ++i)
			sum += vec1[i] * vec2[i];
		return sum;
	}

	template<typename _TYPE>
	inline vector<_TYPE, 3> crossproduct(const vector<_TYPE, 3>& vec1, const vector<_TYPE, 3>& vec2)
	{
		return vector<_TYPE, 3>((vec1.y * vec2.z) - (vec1.z * vec2.y),
								(vec1.z * vec2.x) - (vec1.x * vec2.z),
								(vec1.x * vec2.y) - (vec1.y * vec2.x));
	}

    template<typename _TYPE, unsigned int _DIM>
    inline _TYPE magnitude(const vector<_TYPE, _DIM>& vec)
    {
        return sqrt(dotproduct(vec, vec));
    }

    template<typename _TYPE, unsigned int _DIM>
    inline vector<_TYPE, _DIM> normalize(const vector<_TYPE, _DIM>& vec)
    {
        return vec / magnitude(vec);
    }

	typedef vector<int, 2> vec2i;
	typedef vector<float, 2> vec2f;
	typedef vector<int, 3> vec3i;
	typedef vector<float, 3> vec3f;
	typedef vector<int, 4> vec4i;
	typedef vector<float, 4> vec4f;
}
#endif /* stm_vector_h */
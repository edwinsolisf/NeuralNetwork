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
			std::fill_n(_data, GetSize(), value);
		}

		vector(const vector& other)
		{
			memcpy(_data, other._data, _DIM * sizeof(_TYPE));
		}

		vector(std::initializer_list<_TYPE> list)
		{
			static_assert(list.size() == GetSize());
			std::copy(list.begin(), list.end(), _data);
		}

		vector& operator=(const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(_DIM == vec.GetSize());
			memcpy(_data, vec.GetData(), _DIM * sizeof(_TYPE));
			return *this;
		}

		//Unary Operators
		inline vector operator+() const
		{
			return *this;
		}

		vector operator-() const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = -_data[i];
			return temp;
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
			vector<O_TYPE, _DIM> temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp[i] = O_TYPE(_data[i]);
			return temp;
		}

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(_TYPE))
		{
			for (unsigned int i = 0; i < _DIM; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		vector& SetAll(_TYPE value)
		{
			for (unsigned int i = 0; i < GetSize(); ++i)
				_data[i] = value;
			return *this;
		}

		//Binary Operators
		vector operator+(const vector& other) const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] + other._data[i];
			return temp;
		}


		vector operator-(const vector& other) const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] - other._data[i];
			return temp;
		}

		vector operator*(const vector& other) const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] * other._data[i];
			return temp;
		}

		vector operator/(const vector& other) const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] / other._data[i];
			return temp;
		}

		vector operator+(const _TYPE& other) const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] + other;
			return temp;
		}

		vector operator-(const _TYPE& other) const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] - other;
			return temp;
		}

		vector operator*(const _TYPE& other) const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] * other;
			return temp;
		}

		vector operator/(const _TYPE& other) const
		{
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] / other;
			return temp;
		}

		vector operator+(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(_DIM == vec.GetSize());
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] + vec[i];
			return temp;
		}

		vector operator-(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(_DIM == vec.GetSize());
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] - vec[i];
			return temp;
		}

		vector operator*(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(_DIM == vec.GetSize());
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] * vec[i];
			return temp;
		}

		vector operator/(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(_DIM == vec.GetSize());
			vector temp;
			for (unsigned int i = 0; i < _DIM; ++i)
				temp._data[i] = _data[i] / vec[i];
			return temp;
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

		vector& operator+=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this + vec;
			return *this;
		}

		vector& operator-=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this - vec;
			return *this;
		}

		vector& operator*=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this * vec;
			return *this;
		}

		vector& operator/=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this / vec;
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

		_TYPE DotProduct(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(vec.GetSize() == _DIM)
			_TYPE sum = 0;
			for (unsigned int i = 0; i < _DIM; ++i)
				sum += _data[i] * vec[i];
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

		//Assigment Operator
		vector& operator=(const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(4 == vec.GetSize());
			memcpy(_data, vec.GetData(), 4 * sizeof(_TYPE));
			return *this;
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
			vector<O_TYPE, 4> temp;
			for (unsigned int i = 0; i < 4; ++i)
				temp._data[i] = O_TYPE(_data[i]);
			return temp;
		}

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(_TYPE))
		{
			for (unsigned int i = 0; i < 4; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		vector& SetAll(_TYPE value)
		{
			for (unsigned int i = 0; i < 4; ++i)
				_data[i] = value;
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

		inline vector operator+(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(4 == vec.GetSize());
			return vector(x + vec[0], y + vec[1], z + vec[2], w + vec[3]);
		}

		inline vector operator-(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(4 == vec.GetSize());
			return vector(x - vec[0], y - vec[1], z - vec[2], w - vec[3]);
		}

		inline vector operator*(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(4 == vec.GetSize());
			return vector(x * vec[0], y * vec[1], z * vec[2], w * vec[3]);
		}

		inline vector operator/(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(4 == vec.GetSize());
			return vector(x / vec[0], y / vec[1], z / vec[2], w / vec[3]);
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

		vector& operator+=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this + vec;
			return *this;
		}

		vector& operator-=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this - vec;
			return *this;
		}

		vector& operator*=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this * vec;
			return *this;
		}

		vector& operator/=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this / vec;
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

		inline _TYPE DotProduct(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(vec.GetSize() == 4);
			return (x * vec[0]) + (y * vec[1]) + (z * vec[2]) + (w * vec[3]);
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

		//Assigment Operator
		vector& operator=(const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(3 == vec.GetSize());
			memcpy(_data, vec.GetData(), 3 * sizeof(_TYPE));
			return *this;
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
			vector<O_TYPE, 3> temp;
			for (unsigned int i = 0; i < 3; ++i)
				temp._data[i] = O_TYPE(_data[i]);
			return temp;
		}

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(_TYPE))
		{
			for (unsigned int i = 0; i < 3; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		vector& SetAll(_TYPE value)
		{
			for (unsigned int i = 0; i < 3; ++i)
				_data[i] = value;
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

		inline vector operator+(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(3 == vec.GetSize());
			return vector(x + vec[0], y + vec[1], z + vec[2]);
		}

		inline vector operator-(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(3 == vec.GetSize());
			return vector(x - vec[0], y - vec[1], z - vec[2]);
		}

		inline vector operator*(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(3 == vec.GetSize());
			return vector(x * vec[0], y * vec[1], z * vec[2]);
		}

		inline vector operator/(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(3 == vec.GetSize());
			return vector(x / vec[0], y / vec[1], z / vec[2]);
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


		vector& operator+=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this + vec;
			return *this;
		}

		vector& operator-=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this - vec;
			return *this;
		}

		vector& operator*=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this * vec;
			return *this;
		}

		vector& operator/=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this / vec;
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

		inline _TYPE DotProduct(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(vec.GetSize() == 3);
			return (x * vec[0]) + (y * vec[1]) + (z * vec[2]);
		}

		inline vector CrossProduct(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(vec.GetSize() == 3);
			return vector((y * vec[2]) - (z * vec[1]),
						  (z * vec[0]) - (x * vec[2]),
						  (x * vec[1]) - (y * vec[0]));
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

		//Assigment Operator
		vector& operator=(const dynamic_vector<_TYPE>& vec)
		{
			stm_assert(2 == vec.GetSize());
			memcpy(_data, vec.GetData(), 2 * sizeof(_TYPE));
			return *this;
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
			vector<O_TYPE, 2> temp;
			for (unsigned int i = 0; i < 2; ++i)
				temp._data[i] = O_TYPE(_data[i]);
			return temp;
		}

		//Data manipulation functions
		vector& ApplyToVector(_TYPE(*func)(_TYPE))
		{
			for (unsigned int i = 0; i < 2; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		vector& SetAll(_TYPE value)
		{
			for (unsigned int i = 0; i < 2; ++i)
				_data[i] = value;
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

		inline vector operator+(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(2 == vec.GetSize());
			return vector(x + vec[0], y + vec[1]);
		}

		inline vector operator-(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(2 == vec.GetSize());
			return vector(x - vec[0], y - vec[1]);
		}

		inline vector operator*(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(2 == vec.GetSize());
			return vector(x * vec[0], y * vec[1]);
		}

		inline vector operator/(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(2 == vec.GetSize());
			return vector(x / vec[0], y / vec[1]);
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

		vector& operator+=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this + vec;
			return *this;
		}

		vector& operator-=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this - vec;
			return *this;
		}

		vector& operator*=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this * vec;
			return *this;
		}

		vector& operator/=(const dynamic_vector<_TYPE>& vec)
		{
			*this = *this / vec;
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

		inline _TYPE DotProduct(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(vec.GetSize() == 2);
			return (x * vec[0]) + (y * vec[1]);
		}

		inline vector<_TYPE, 3> CrossProduct(const dynamic_vector<_TYPE>& vec) const
		{
			stm_assert(vec.GetSize() == 2);
			return vector(0, 0, (x * vec[1]) - (y * vec[0]));
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

	template<typename _TYPE, unsigned int _DIM>
	inline vector<_TYPE, _DIM> pow(const vector<_TYPE, _DIM>& vec, unsigned int power)
	{
		vector<_TYPE, _DIM> out = vec;
		for (unsigned int i = 0; i < (power - 1); ++i)
			out *= vec;
		return out;
	}

	typedef vector<int, 2> vec2i;
	typedef vector<float, 2> vec2f;
	typedef vector<int, 3> vec3i;
	typedef vector<float, 3> vec3f;
	typedef vector<int, 4> vec4i;
	typedef vector<float, 4> vec4f;
}
#endif /* stm_vector_h */
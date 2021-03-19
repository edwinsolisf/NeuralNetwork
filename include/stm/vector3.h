#ifndef stm_vector_3_h
#define stm_vector_3_h

namespace stm
{
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

	typedef vector<int, 3> vec3i;
	typedef vector<float, 3> vec3f;
}

#endif /* stm_vector_3_h */

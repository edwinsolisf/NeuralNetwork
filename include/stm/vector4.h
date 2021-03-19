#ifndef stm_vector_4_h
#define stm_vector_4_h

namespace stm
{
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

	typedef vector<int, 4> vec4i;
	typedef vector<float, 4> vec4f;
}

#endif /* stm_vector_4_h */

#ifndef stm_vector_2_h
#define stm_vector_2_h

namespace stm
{
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

	typedef vector<int, 2> vec2i;
	typedef vector<float, 2> vec2f;
}

#endif /* stm_vector_2_h */

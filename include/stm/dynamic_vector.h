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
		dynamic_vector(unsigned int dimensions)
			:_data(new _T[dimensions]{ 0 }), _dimensions(dimensions)
		{
			stm_assert(dimensions != 0);
		}

		dynamic_vector(unsigned int dimensions, const _T* data)
			:_data(new _T[dimensions]), _dimensions(dimensions)
		{
			stm_assert(dimensions != 0);
			memcpy(_data, data, _dimensions * sizeof(_T));
		}

		dynamic_vector(unsigned int dimensions, const _T& value)
			:_data(new _T[dimensions]{ value }), _dimensions(dimensions)
		{
			stm_assert(dimensions != 0);
		}

		dynamic_vector(const dynamic_vector& other)
			:_data(new _T[other._dimensions]), _dimensions(other._dimensions)
		{
			memcpy(_data, other._data, _dimensions * sizeof(_T));
		}

		dynamic_vector(dynamic_vector&& other) noexcept
			:_data(std::exchange(other._data, nullptr)), _dimensions(std::exchange(other._dimensions, 0))
		{
		}

		template<unsigned int dimensions>
		dynamic_vector(const vector<_T, dimensions>& static_vector)
			: _data(new _T[dimensions]), _dimensions(dimensions)
		{
			memcpy(_data, static_vector.GetData(), dimensions * sizeof(_T));
		}

		//Assignment operators
		dynamic_vector& operator=(const dynamic_vector& other)
		{
			if (this == &other) { return *this; }
			if (_dimensions == other._dimensions)
				memcpy(_data, other._data, _dimensions * sizeof(_T));
			else
			{
				_T* newData = new _T[other._dimensions];
				memcpy(newData, other._data, other._dimensions * sizeof(_T));
				delete _data;
				_data = newData;
			}
			_dimensions = other._dimensions;

			return *this;
		}

		dynamic_vector& operator=(dynamic_vector&& other) noexcept
		{
			if (this == &other) { return *this; }
			std::swap(_data, other._data);
			std::swap(_dimensions, other._dimensions);

			return *this;
		}

		//Unary operators
		inline dynamic_vector operator+() const
		{
			return *this;
		}

		dynamic_vector operator-() const
		{
			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = -_data[i];
			return std::move(temp);
		}

		inline _T& operator[](unsigned int index) { stm_assert(index < _dimensions); return _data[index]; }
		inline const _T& operator[](unsigned int index) const { stm_assert(index < _dimensions); return _data[index]; }

		//Data Manipulation Functions
		void Resize(unsigned int dimensions)
		{
			stm_assert(dimensions != 0);
			if (dimensions > GetSize())
			{
				_T* newData = new _T[dimensions]{ 0 };
				memcpy(newData, _data, _dimensions * sizeof(_T));
				delete[] _data;
				_data = newData;
			}
		}

		dynamic_vector& ApplyToVector(_T(*func)(_T))
		{
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = func(_data[i]);
			return *this;
		}

		//Binary Operators
		dynamic_vector operator+(const dynamic_vector& other) const
		{
			stm_assert(_dimensions == other._dimensions);

			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] + other[i];
			return std::move(temp);
		}

		dynamic_vector operator-(const dynamic_vector& other) const
		{
			stm_assert(_dimensions == other._dimensions);

			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] - other[i];
			return std::move(temp);
		}

		dynamic_vector operator*(const dynamic_vector& other) const
		{
			stm_assert(_dimensions == other._dimensions);

			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] * other[i];
			return std::move(temp);
		}

		dynamic_vector operator/(const dynamic_vector& other) const
		{
			stm_assert(_dimensions == other._dimensions);

			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] / other[i];
			return std::move(temp);
		}

		template<unsigned int dimensions>
		vector<_T, dimensions> operator+(const vector<_T, dimensions>& static_vector) const
		{
			stm_assert(_dimensions == dimensions);
			_T temp[dimensions];
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] + static_vector[i];
			return vector<_T, dimensions>(temp);
		}

		template<unsigned int dimensions>
		vector<_T, dimensions> operator-(const vector<_T, dimensions>& static_vector) const
		{
			stm_assert(_dimensions == dimensions);
			_T temp[dimensions];
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] - static_vector[i];
			return vector<_T, dimensions>(temp);
		}

		template<unsigned int dimensions>
		vector<_T, dimensions> operator*(const vector<_T, dimensions>& static_vector) const
		{
			stm_assert(_dimensions == dimensions);
			_T temp[dimensions];
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] * static_vector[i];
			return vector<_T, dimensions>(temp);
		}

		template<unsigned int dimensions>
		vector<_T, dimensions> operator/(const vector<_T, dimensions>& static_vector) const
		{
			stm_assert(_dimensions == dimensions);
			_T temp[dimensions];
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] / static_vector[i];
			return vector<_T, dimensions>(temp);
		}

		dynamic_vector operator+(const _T& value) const
		{
			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] + value;
			return std::move(temp);
		}

		dynamic_vector operator-(const _T& value) const
		{
			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] - value;
			return std::move(temp);
		}

		dynamic_vector operator*(const _T& value) const
		{
			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] * value;
			return std::move(temp);
		}

		dynamic_vector operator/(const _T& value) const
		{
			dynamic_vector temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp[i] = _data[i] / value;
			return std::move(temp);
		}

		//Binary Assigment Operators
		dynamic_vector& operator+=(const dynamic_vector& other)
		{
			stm_assert(_dimensions == other._dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] + other[i];
			return *this;
		}

		dynamic_vector& operator-=(const dynamic_vector& other)
		{
			stm_assert(_dimensions == other._dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] - other[i];
			return *this;
		}

		dynamic_vector& operator*=(const dynamic_vector& other)
		{
			stm_assert(_dimensions == other._dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] * other[i];
			return *this;
		}

		dynamic_vector& operator/=(const dynamic_vector& other)
		{
			stm_assert(_dimensions == other._dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] / other[i];
			return *this;
		}

		template<unsigned int dimensions>
		dynamic_vector& operator+=(const vector<_T, dimensions>& static_vector)
		{
			stm_assert(_dimensions == dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] + static_vector[i];
			return *this;
		}

		template<unsigned int dimensions>
		dynamic_vector& operator-=(const vector<_T, dimensions>& static_vector)
		{
			stm_assert(_dimensions == dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] - static_vector[i];
			return *this;
		}

		template<unsigned int dimensions>
		dynamic_vector& operator*=(const vector<_T, dimensions>& static_vector)
		{
			stm_assert(_dimensions == dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] * static_vector[i];
			return *this;
		}

		template<unsigned int dimensions>
		dynamic_vector& operator/=(const vector<_T, dimensions>& static_vector)
		{
			stm_assert(_dimensions == dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] / static_vector[i];
			return *this;
		}

		dynamic_vector& operator+=(const _T& value)
		{
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] + value;
			return *this;
		}

		dynamic_vector& operator-=(const _T& value)
		{
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] - value;
			return *this;
		}

		dynamic_vector& operator*=(const _T& value)
		{
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] * value;
			return *this;
		}

		dynamic_vector& operator/=(const _T& value)
		{
			for (unsigned int i = 0; i < _dimensions; ++i)
				_data[i] = _data[i] / value;
			return *this;
		}
		//Casting
		template<typename O_TYPE>
		dynamic_vector<O_TYPE> Cast() const
		{
			dynamic_vector<O_TYPE> temp(_dimensions);
			for (unsigned int i = 0; i < _dimensions; ++i)
				temp._data[i] = O_TYPE(_data[i]);
			return std::move(temp);
		}

		//Data Info Functions
		inline _T* GetData() { return _data; }
		inline const _T* GetData() const { return _data; }
		inline unsigned int GetSize() const { return _dimensions; }

		//Math Functions
		inline _T Magnitude() const
		{
			return sqrt(DotProduct(*this));
		}

		inline dynamic_vector UnitVector() const
		{
			return std::move((*this) / Magnitude());
		}

		_T DotProduct(const dynamic_vector& other) const
		{
			stm_assert(_dimensions == other._dimensions)
			_T sum = 0;
			for (unsigned int i = 0; i < _dimensions; ++i)
				sum += _data[i] * other[i];
			return sum;
		}

		template<unsigned int dimensions>
		_T DotProduct(const vector<_T, dimensions>& other) const
		{
			stm_assert(_dimensions == dimensions);
			_T sum = 0;
			for (unsigned int i = 0; i < _dimensions; ++i)
				sum += _data[i] * other[i];
			return sum;
		}

		inline vector<_T, 3> CrossProduct(const dynamic_vector<_T>& vec)
		{
			stm_assert(_dimensions == 3 && vec.GetSize() == 3);
			return vector<_T, 3>((_data[1] * vec[2]) - (_data[2] * vec[1]),
									(_data[2] * vec[0]) - (_data[0] * vec[2]),
									(_data[0] * vec[1]) - (_data[1] * vec[0]));
		}

		inline vector<_T, 3> CrossProduct(const vector<_T, 3>& vec)
		{
			stm_assert(_dimensions == 3 && vec.GetSize() == 3);
			return vector<_T, 3>((_data[1] * vec[2]) - (_data[2] * vec[1]),
									(_data[2] * vec[0]) - (_data[0] * vec[2]),
									(_data[0] * vec[1]) - (_data[1] * vec[0]));
		}

		//Destructors
		~dynamic_vector()
		{
			delete _data;
		}

	private:
		_T* _data;
		unsigned int _dimensions;
	};

	template<typename _TYPE>
	_TYPE dotproduct(const dynamic_vector<_TYPE>& vec1, const dynamic_vector<_TYPE>& vec2)
	{
		stm_assert(vec1.GetSize() == vec2.GetSize());
		_TYPE sum = 0;
		for (unsigned int i = 0; i < vec1.GetSize(); ++i)
			sum += vec1[i] * vec2[i];
		return sum;
	}

	template<typename _TYPE, unsigned int _DIM>
	_TYPE dotproduct(const vector<_TYPE, _DIM>& vec1, const dynamic_vector<_TYPE>& vec2)
	{
		stm_assert(vec1.GetSize() == vec2.GetSize());
		_TYPE sum = 0;
		for (unsigned int i = 0; i < vec1.GetSize(); ++i)
			sum += vec1[i] * vec2[i];
		return sum;
	}

	template<typename _TYPE, unsigned int _DIM>
	_TYPE dotproduct(const dynamic_vector<_TYPE>& vec1, const vector<_TYPE, _DIM>& vec2)
	{
		stm_assert(vec1.GetSize() == vec2.GetSize());
		_TYPE sum = 0;
		for (unsigned int i = 0; i < vec1.GetSize(); ++i)
			sum += vec1[i] * vec2[i];
		return sum;
	}

	template<typename _TYPE>
	inline vector<_TYPE, 3> crossproduct(const dynamic_vector<_TYPE>& vec1, const dynamic_vector<_TYPE>& vec2)
	{
		stm_assert(vec1.GetSize() == 3 && vec2.GetSize() == 3);
		return vector<_TYPE, 3>((vec1[1] * vec2[2]) - (vec1[2] * vec2[1]),
								(vec1[2] * vec2[0]) - (vec1[0] * vec2[2]),
								(vec1[0] * vec2[1]) - (vec1[1] * vec2[0]));
	}

	template<typename _TYPE>
	inline vector<_TYPE, 3> crossproduct(const vector<_TYPE, 3>& vec1, const dynamic_vector<_TYPE>& vec2)
	{
		stm_assert(vec1.GetSize() == 3 && vec2.GetSize() == 3);
		return vector<_TYPE, 3>((vec1[1] * vec2[2]) - (vec1[2] * vec2[1]),
								(vec1[2] * vec2[0]) - (vec1[0] * vec2[2]),
								(vec1[0] * vec2[1]) - (vec1[1] * vec2[0]));
	}

	template<typename _TYPE>
	inline vector<_TYPE, 3> crossproduct(const dynamic_vector<_TYPE>& vec1, const vector<_TYPE, 3>& vec2)
	{
		stm_assert(vec1.GetSize() == 3 && vec2.GetSize() == 3);
		return vector<_TYPE, 3>((vec1[1] * vec2[2]) - (vec1[2] * vec2[1]),
								(vec1[2] * vec2[0]) - (vec1[0] * vec2[2]),
								(vec1[0] * vec2[1]) - (vec1[1] * vec2[0]));
	}

	template<typename _TYPE>
	inline _TYPE magnitude(const dynamic_vector<_TYPE>& vec)
	{
		return sqrt(dotproduct(vec, vec));
	}

	template<typename _TYPE>
	inline dynamic_vector<_TYPE> normalize(const dynamic_vector<_TYPE>& vec)
	{
		return vec / magnitude(vec);
	}

	typedef dynamic_vector<float> vec_f;
	typedef dynamic_vector<int> vec_i;
}

#endif /* stm_dynamic_vector_h */

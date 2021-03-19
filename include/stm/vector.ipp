namespace stm
{
	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>::vector()
	{
		memset(_data, 0, _DIM * sizeof(_TYPE));
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>::vector(const _TYPE data[_DIM])
	{
		memcpy(_data, data, _DIM * sizeof(_TYPE));
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>::vector(const _TYPE& value)
	{
		std::fill_n(_data, GetSize(), value);
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>::vector(const vector& other)
	{
		memcpy(_data, other._data, _DIM * sizeof(_TYPE));
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>::vector(std::initializer_list<_TYPE> list)
	{
		static_assert(list.size() == GetSize());
		std::copy(list.begin(), list.end(), _data);
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator=(const dynamic_vector<_TYPE>& vec)
	{
		stm_assert(_DIM == vec.GetSize());
		memcpy(_data, vec.GetData(), _DIM * sizeof(_TYPE));
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator-() const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = -_data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM> template<typename O_TYPE>
	vector<O_TYPE, _DIM> vector<_TYPE, _DIM>::Cast() const
	{
		vector<O_TYPE, _DIM> temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp[i] = O_TYPE(_data[i]);
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::ApplyToVector(_TYPE(*func)(_TYPE))
	{
		for (unsigned int i = 0; i < _DIM; ++i)
			_data[i] = func(_data[i]);
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::SetAll(_TYPE value)
	{
		for (unsigned int i = 0; i < GetSize(); ++i)
			_data[i] = value;
		return *this;
	}
	
	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator+(const vector& other) const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] + other._data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator-(const vector& other) const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] - other._data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator*(const vector& other) const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] * other._data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator/(const vector& other) const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] / other._data[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator+(const _TYPE& other) const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] + other;
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator-(const _TYPE& other) const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] - other;
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator*(const _TYPE& other) const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] * other;
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator/(const _TYPE& other) const
	{
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] / other;
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator+(const dynamic_vector<_TYPE>& vec) const
	{
		stm_assert(_DIM == vec.GetSize());
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] + vec[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator-(const dynamic_vector<_TYPE>& vec) const
	{
		stm_assert(_DIM == vec.GetSize());
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] - vec[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator*(const dynamic_vector<_TYPE>& vec) const
	{
		stm_assert(_DIM == vec.GetSize());
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] * vec[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> vector<_TYPE, _DIM>::operator/(const dynamic_vector<_TYPE>& vec) const
	{
		stm_assert(_DIM == vec.GetSize());
		vector temp;
		for (unsigned int i = 0; i < _DIM; ++i)
			temp._data[i] = _data[i] / vec[i];
		return temp;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator+=(const vector& other)
	{
		*this = *this + other;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator-=(const vector& other)
	{
		*this = *this - other;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator*=(const vector& other)
	{
		*this = *this * other;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator/=(const vector& other)
	{
		*this = *this / other;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator+=(const _TYPE& other)
	{
		*this = *this + other;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator-=(const _TYPE& other)
	{
		*this = *this - other;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator*=(const _TYPE& other)
	{
		*this = *this * other;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator/=(const _TYPE& other)
	{
		*this = *this / other;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator+=(const dynamic_vector<_TYPE>& vec)
	{
		*this = *this + vec;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator-=(const dynamic_vector<_TYPE>& vec)
	{
		*this = *this - vec;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator*=(const dynamic_vector<_TYPE>& vec)
	{
		*this = *this * vec;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM>& vector<_TYPE, _DIM>::operator/=(const dynamic_vector<_TYPE>& vec)
	{
		*this = *this / vec;
		return *this;
	}

	template<typename _TYPE, unsigned int _DIM>
	_TYPE vector<_TYPE, _DIM>::DotProduct(const vector& other) const
	{
		_TYPE sum = 0;
		for (unsigned int i = 0; i < _DIM; ++i)
			sum += _data[i] * other[i];
		return sum;
	}

	template<typename _TYPE, unsigned int _DIM>
	_TYPE vector<_TYPE, _DIM>::DotProduct(const dynamic_vector<_TYPE>& vec) const
	{
		stm_assert(vec.GetSize() == _DIM)
			_TYPE sum = 0;
		for (unsigned int i = 0; i < _DIM; ++i)
			sum += _data[i] * vec[i];
		return sum;
	}

	template<typename _TYPE, unsigned int _DIM>
	_TYPE dotproduct(const vector<_TYPE, _DIM>& vec1, const vector<_TYPE, _DIM>& vec2)
	{
		_TYPE sum = 0;
		for (unsigned int i = 0; i < _DIM; ++i)
			sum += vec1[i] * vec2[i];
		return sum;
	}

	template<typename _TYPE, unsigned int _DIM>
	vector<_TYPE, _DIM> pow(const vector<_TYPE, _DIM>& vec, unsigned int power)
	{
		vector<_TYPE, _DIM> out = vec;
		for (unsigned int i = 0; i < (power - 1); ++i)
			out *= vec;
		return out;
	}
}
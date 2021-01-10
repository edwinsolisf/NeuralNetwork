namespace stm
{
	template<typename _T>
	dynamic_vector<_T>::dynamic_vector(unsigned int dimensions)
		:_data(new _T[dimensions]), _dimensions(dimensions)
	{
		memset(_data, 0, dimensions * sizeof(_T));
		stm_assert(dimensions != 0);
	}

	template<typename _T>
	dynamic_vector<_T>::dynamic_vector(unsigned int dimensions, const _T* data)
		:_data(new _T[dimensions]), _dimensions(dimensions)
	{
		stm_assert(dimensions != 0);
		memcpy(_data, data, _dimensions * sizeof(_T));
	}

	template<typename _T>
	dynamic_vector<_T>::dynamic_vector(unsigned int dimensions, _T value)
		:_data(new _T[dimensions]), _dimensions(dimensions)
	{
		stm_assert(dimensions != 0);
		std::fill_n(_data, _dimensions, value);
	}

	template<typename _T>
	dynamic_vector<_T>::dynamic_vector(unsigned int dimensions, std::initializer_list<_T> list)
	{
		stm_assert(list.size() == _dimensions);
		std::copy(list.begin(), list.end(), _data);
	}

	template<typename _T>
	dynamic_vector<_T>::dynamic_vector(const dynamic_vector& other)
		:_data(new _T[other._dimensions]), _dimensions(other._dimensions)
	{
		memcpy(_data, other._data, _dimensions * sizeof(_T));
	}

	template<typename _T>
	dynamic_vector<_T>::dynamic_vector(dynamic_vector&& other) noexcept
		:_data(std::exchange(other._data, nullptr)), _dimensions(std::exchange(other._dimensions, 0))
	{
	}

	template<typename _T>
	dynamic_vector<_T>::dynamic_vector(_T*& data, unsigned int dimensions)
		:_data(std::exchange(data, nullptr)), _dimensions(dimensions)
	{
		stm_assert(_data != nullptr);
	}

	template<typename _T> template<unsigned int dimensions>
	dynamic_vector<_T>::dynamic_vector(const vector<_T, dimensions>& static_vector)
		: _data(new _T[dimensions]), _dimensions(dimensions)
	{
		memcpy(_data, static_vector.GetData(), dimensions * sizeof(_T));
	}

	template<typename _T>
	dynamic_vector<_T>::~dynamic_vector()
	{
		delete _data;
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator=(const dynamic_vector& other)
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

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator=(dynamic_vector&& other) noexcept
	{
		if (this == &other) { return *this; }
		std::swap(_data, other._data);
		std::swap(_dimensions, other._dimensions);

		return *this;
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator-() const&
	{
		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = -_data[i];
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator-() &&
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = -_data[i];
		return std::move(*this);
	}

	template<typename _T>
	void dynamic_vector<_T>::Resize(unsigned int dimensions)
	{
		stm_assert(dimensions != 0);
		if (dimensions > GetSize())
		{
			_T* newData = new _T[dimensions];
			memset(newData, 0, sizeof(_T) * dimensions);
			memcpy(newData, _data, _dimensions * sizeof(_T));
			delete[] _data;
			_data = newData;
		}
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::SetAll(_T value) &
	{
		for (unsigned int i = 0; i < GetSize(); ++i)
			_data[i] = value;
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::SetAll(_T value) &&
	{
		for (unsigned int i = 0; i < GetSize(); ++i)
			_data[i] = value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::ApplyToVector(_T(*func)(_T)) &
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = func(_data[i]);
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::ApplyToVector(_T(*func)(_T)) &&
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = func(_data[i]);
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator+(const dynamic_vector& other) const&
	{
		stm_assert(_dimensions == other._dimensions);

		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] + other[i];
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator+(const dynamic_vector& other) &&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] += other[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator+(dynamic_vector&& other) const&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			other[i] = _data[i] + other[i];
		return std::move(other);
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator+(dynamic_vector&& other)&&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] += other[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator-(const dynamic_vector& other) const&
	{
		stm_assert(_dimensions == other._dimensions);

		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] - other[i];
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator-(const dynamic_vector& other) &&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] -= other[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator-(dynamic_vector&& other) const&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			other[i] = _data[i] - other[i];
		return std::move(other);
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator-(dynamic_vector&& other) &&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] -= other[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator*(const dynamic_vector& other) const&
	{
		stm_assert(_dimensions == other._dimensions);
		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] * other[i];
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator*(const dynamic_vector& other) &&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] *= other[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator*(dynamic_vector&& other) const&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			other[i] = _data[i] * other[i];
		return std::move(other);
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator*(dynamic_vector&& other) &&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] *= other[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator/(const dynamic_vector& other) const&
	{
		stm_assert(_dimensions == other._dimensions);
		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] / other[i];
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator/(const dynamic_vector& other)&&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] /= other[i];
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator/(dynamic_vector&& other) const&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			other[i] = _data[i] / other[i];
		return std::move(other);
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator/(dynamic_vector&& other) &&
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] /= other[i];
		return std::move(*this);
	}

	template<typename _T> template<unsigned int dimensions>
	vector<_T, dimensions> dynamic_vector<_T>::operator+(const vector<_T, dimensions>& static_vector) const
	{
		stm_assert(_dimensions == dimensions);
		vector<_T, dimensions> temp;
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] + static_vector[i];
		return temp;
	}

	template<typename _T> template<unsigned int dimensions>
	vector<_T, dimensions> dynamic_vector<_T>::operator-(const vector<_T, dimensions>& static_vector) const
	{
		stm_assert(_dimensions == dimensions);
		vector<_T, dimensions> temp;
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] - static_vector[i];
		return temp;
	}

	template<typename _T> template<unsigned int dimensions>
	vector<_T, dimensions> dynamic_vector<_T>::operator*(const vector<_T, dimensions>& static_vector) const
	{
		stm_assert(_dimensions == dimensions);
		vector<_T, dimensions> temp;
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] * static_vector[i];
		return temp;
	}

	template<typename _T> template<unsigned int dimensions>
	vector<_T, dimensions> dynamic_vector<_T>::operator/(const vector<_T, dimensions>& static_vector) const
	{
		stm_assert(_dimensions == dimensions);
		vector<_T, dimensions> temp;
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] / static_vector[i];
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator+(const _T& value) const&
	{
		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] + value;
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator+(const _T& value)&&
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] += value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator-(const _T& value) const&
	{
		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] - value;
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator-(const _T& value) &&
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] -= value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator*(const _T& value) const&
	{
		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] * value;
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator*(const _T& value)&&
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] *= value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T> dynamic_vector<_T>::operator/(const _T& value) const&
	{
		dynamic_vector temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp[i] = _data[i] / value;
		return temp;
	}

	template<typename _T>
	dynamic_vector<_T>&& dynamic_vector<_T>::operator/(const _T& value)&&
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] /= value;
		return std::move(*this);
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator+=(const dynamic_vector& other)
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] + other[i];
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator-=(const dynamic_vector& other)
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] - other[i];
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator*=(const dynamic_vector& other)
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] * other[i];
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator/=(const dynamic_vector& other)
	{
		stm_assert(_dimensions == other._dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] / other[i];
		return *this;
	}

	template<typename _T> template<unsigned int dimensions>
	dynamic_vector<_T>& dynamic_vector<_T>::operator+=(const vector<_T, dimensions>& static_vector)
	{
		stm_assert(_dimensions == dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] + static_vector[i];
		return *this;
	}

	template<typename _T> template<unsigned int dimensions>
	dynamic_vector<_T>& dynamic_vector<_T>::operator-=(const vector<_T, dimensions>& static_vector)
	{
		stm_assert(_dimensions == dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] - static_vector[i];
		return *this;
	}

	template<typename _T> template<unsigned int dimensions>
	dynamic_vector<_T>& dynamic_vector<_T>::operator*=(const vector<_T, dimensions>& static_vector)
	{
		stm_assert(_dimensions == dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] * static_vector[i];
		return *this;
	}

	template<typename _T> template<unsigned int dimensions>
	dynamic_vector<_T>& dynamic_vector<_T>::operator/=(const vector<_T, dimensions>& static_vector)
	{
		stm_assert(_dimensions == dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] / static_vector[i];
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator+=(const _T& value)
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] + value;
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator-=(const _T& value)
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] - value;
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator*=(const _T& value)
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] * value;
		return *this;
	}

	template<typename _T>
	dynamic_vector<_T>& dynamic_vector<_T>::operator/=(const _T& value)
	{
		for (unsigned int i = 0; i < _dimensions; ++i)
			_data[i] = _data[i] / value;
		return *this;
	}

	template<typename _T> template<typename O_TYPE>
	dynamic_vector<O_TYPE> dynamic_vector<_T>::Cast() const
	{
		dynamic_vector<O_TYPE> temp(_dimensions);
		for (unsigned int i = 0; i < _dimensions; ++i)
			temp._data[i] = O_TYPE(_data[i]);
		return temp;
	}

	template<typename _T>
	_T dynamic_vector<_T>::DotProduct(const dynamic_vector& other) const
	{
		stm_assert(_dimensions == other._dimensions)
			_T sum = 0;
		for (unsigned int i = 0; i < _dimensions; ++i)
			sum += _data[i] * other[i];
		return sum;
	}

	template<typename _T> template<unsigned int dimensions>
	_T dynamic_vector<_T>::DotProduct(const vector<_T, dimensions>& other) const
	{
		stm_assert(_dimensions == dimensions);
		_T sum = 0;
		for (unsigned int i = 0; i < _dimensions; ++i)
			sum += _data[i] * other[i];
		return sum;
	}

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
}

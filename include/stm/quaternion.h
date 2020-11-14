#pragma once
#include <iostream>
#include "vector.h"
#include <math.h>

namespace stml
{
	template<typename _TYPE>
	class quaternion
	{
	public:
		struct
		{
			union
			{
				_TYPE _data[4];
				struct
				{
					_TYPE r, i, j, k;
				};
			};
		};

		quaternion()
			:r(0), i(0), j(0), k(0)
		{
		}
        
		quaternion(const _TYPE data[4])
			:r(_data[0]), i(_data[1]), j(_data[2]), k(_data[3])
		{
		}

        quaternion(_TYPE value)
        :r(value), i(value), j(value), k(value)
        {
        }
        
        quaternion(_TYPE scalar, const vector<_TYPE, 3>& vector)
        :r(scalar), i(vector.x), j(vector.y), k(vector.z)
        {
            
        }
        
        quaternion(const vector<_TYPE, 4>& values)
        {
            memcpy(_data, values.GetData(), 4 * sizeof(_TYPE));
        }
        
		quaternion(const _TYPE& rPart, const _TYPE& iPart, const _TYPE& jPart, const _TYPE& kPart)
			:r(rPart), i(iPart), j(jPart), k(kPart)
		{
		}

		inline quaternion operator+(const quaternion& other) const
		{
			return quaternion(r + other.r, i + other.i, j + other.j, k + other.k);
		}

		inline quaternion operator-(const quaternion& other) const
		{
			return quaternion(r - other.r, i - other.i, j - other.j, k - other.k);
		}

		quaternion operator*(const quaternion& other) const
		{
			return quaternion((r * other.r) - (i * other.i) - (j * other.j) - (k * other.k),
                              (r * other.i) + (i * other.r) + (j * other.k) - (k * other.j),
                              (r * other.j) - (i * other.k) + (j * other.r) + (k * other.i),
                              (r * other.k) + (i * other.j) - (j * other.i) + (k * other.r));
		}
        
        inline quaternion operator/(const quaternion& other) const
        {
            return Reciprocal() * other;
        }
        
        inline quaternion operator+(_TYPE other) const
        {
            return quaternion(r + other, i, j, k);
        }
        
        inline quaternion operator-(_TYPE other) const
        {
            return quaternion(r - other, i, j, k);
        }
        
        inline quaternion operator*(_TYPE other) const
        {
            return quaternion(r * other, i * other, j * other, k * other);
        }
        
        inline quaternion operator/(_TYPE other) const
        {
            return quaternion(r / other, i / other, j / other, k / other);
        }
        
		inline quaternion Conjugate() const
		{
			return quaternion(r, -i, -j, -k);
		}

		inline _TYPE Magnitude() const
		{
			return sqrtf((r * r) + (i * i) + (j * j) + (k * k));
		}
        
        inline quaternion Reciprocal() const
        {
            return Conjugate() / Magnitude();
        }
        
        inline quaternion Versor() const
        {
            return (*this) / Magnitude();
        }
        
        inline _TYPE RealPart() const
        {
            return r;
        }
        
        inline vector<_TYPE, 3> VectorPart() const
        {
            return vector<_TYPE, 3>(&_data[1]);
        }
        
        inline vector<_TYPE, 4> ToVector() const
        {
            return vector<_TYPE, 4>(_data);
        }
	};

    typedef quaternion<float> quatf;
    typedef quaternion<int>   quati;
}

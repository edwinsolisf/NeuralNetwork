#ifndef stm_complex_h
#define stm_complex_h

#include <iostream>
#include <math.h>
#include <cmath>

namespace stm
{
	template<typename _TYPE>
	class complex
	{
	public:
		struct
		{
			union
			{
				_TYPE data[2];
				struct
				{
					_TYPE r, i;
				};
			};
		};

		complex()
			:r(0), i(0)
		{
		}

		complex(const _TYPE& rPart, const _TYPE& iPart)
			:r(rPart), i(iPart)
		{
		}

		complex(const _TYPE data[2])
			:r(data[0]), i(data[1])
		{
		}

		complex operator+(const complex& other) const
		{
			return complex(r + other.r, i + other.i);
		}

		complex operator+(const complex& other) const
		{
			return complex(r + other.r, i + other.i);
		}

		complex operator-(const complex& other) const
		{
			return complex(r - other.r, i - other.i);
		}

		complex operator+(const _TYPE& other) const
		{
			return complex(r + other, i);
		}

		complex operator-(const _TYPE& other) const
		{
			return complex(r - other, i);
		}

		complex operator*(const _TYPE& other) const
		{
			return complex(r * other, i * other);
		}

		complex operator/(const _TYPE& other) const
		{
			return complex(r / other, i / other);
		}

		complex operator*(const complex& other) const
		{
			return complex((r * other.r) - (i * other.i), (i * other.r) + (r * other.i));
		}

		complex operator/(const complex& other) const
		{
			return (*this) * other.Conjugate() / (other.Magnitude() * other.Magnitude);
		}

		inline complex Conjugate() const
		{
			return complex(r, -i);
		}

		inline _TYPE Magnitude() const
		{
			return sqrt((r * r) + (i * i));
		}

		inline _TYPE Direction() const
		{
			return (i > 0 && r < 0) ? (atan(1) * 2 + atan(i / r)) : atan(i / r);
		}
	};
}
#endif /* stm_complex_h */
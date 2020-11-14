#pragma once

#include "matrix.h"
#include "vector.h"
#include "utilities.h"
#include "constants.h"

namespace stml
{
	inline float radians(const float& angleInDegs) { return angleInDegs * PI_f / 180.0f; }
	inline double radians(const double& angleInDegs) { return angleInDegs * PI / 180.0; }

	inline float degrees(const float& angleInRads) { return angleInRads * 180.0f / PI_f; }
	inline double degrees(const double& angleInRads) { return angleInRads * 180.0 / PI; }
}

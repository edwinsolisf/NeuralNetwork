#ifndef stm_matrix_transform_h
#define stm_matrix_transform_h

#include "vector.h"
#include "matrix.h"

namespace stm
{

	const vec3f posX = { 1.0f, 0.0f, 0.0f };
	const vec3f negX = {-1.0f, 0.0f, 0.0f };
	const vec3f posY = { 0.0f, 1.0f, 0.0f };
	const vec3f negY = { 1.0f,-1.0f, 0.0f };
	const vec3f posZ = { 1.0f, 0.0f, 1.0f };
	const vec3f negZ = { 1.0f, 0.0f,-1.0f };

	mat4f scale(const vec3f& scaleFactor);
	mat4f scale(const float& xScale, const float& yScale, const float& zScale);

	mat4f translate(const vec3f& translateFactor);
	mat4f translate(const float& xTrans, const float& yTrans, const float& zTrans);

	mat4f rotateX(const float& angleInRads);
	mat4f rotateY(const float& angleInRads);
	mat4f rotateZ(const float& angleInRads);
	mat4f rotate(const vec3f& axis, const float& angleInRads);
    vec3f rotation(const vec3f& vec, const vec3f& axis, float angle);

	mat4f ortho(const float& xleft, const float& xright, const float& ybottom, const float& ytop, const float& znear, const float& zfar);
	mat4f perspective(const float& FOVRads, const float& aspectRatio, const float& zNear, const float& zFar);
    mat4f lookAt(const vec3f& position, const vec3f& up, const vec3f& right);
    mat4f lookAt(const vec3f& position, const vec3f& up, const vec3f& right, const vec3f& direction);

    float angle(const vec3f& vec1, const vec3f& vec2);
}
#endif /* stm_matrix_transform_h */
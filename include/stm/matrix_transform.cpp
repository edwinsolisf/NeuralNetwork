#include "matrix_transform.h"
#include "constants.h"
#include "quaternion.h"

namespace stml
{
	mat4f scale(const vec3f& scaleFactor)
	{
		mat4f temp;
		temp[0][0] = scaleFactor.x;
		temp[1][1] = scaleFactor.y;
		temp[2][2] = scaleFactor.z;
		temp[3][3] = 1.0f;

		return temp;
	}

	mat4f scale(const float& xScale, const float& yScale, const float& zScale)
	{
		mat4f temp;
		temp[0][0] = xScale;
		temp[1][1] = yScale;
		temp[2][2] = zScale;
		temp[3][3] = 1.0f;
		
		return temp;
	}

	mat4f translate(const vec3f& translateFactor)
	{
		mat4f temp = identity_mat4f;
		temp[0][3] = translateFactor.x;
		temp[1][3] = translateFactor.y;
		temp[2][3] = translateFactor.z;
        temp[3][3] = 1.0f;
		
        return temp;
	}

	mat4f translate(const float& xTrans, const float& yTrans, const float& zTrans)
	{
		mat4f temp = identity_mat4f;
		temp[0][3] = xTrans;
		temp[1][3] = yTrans;
		temp[2][3] = zTrans;
        temp[3][3] = 1.0f;
		
        return temp;
	}

	mat4f rotateX(const float& angleInRads)
	{
		mat4f temp = identity_mat4f;
		temp[1][1] = cos(angleInRads);
		temp[1][2] = -sin(angleInRads);
		temp[2][1] = sin(angleInRads);
		temp[2][2] = cos(angleInRads);
        temp[3][3] = 1.0f;
		
        return temp;
	}

	mat4f rotateY(const float& angleInRads)
	{
		mat4f temp = identity_mat4f;
		temp[0][0] = cos(angleInRads);
		temp[0][2] = sin(angleInRads);
		temp[2][0] = -sin(angleInRads);
		temp[2][2] = cos(angleInRads);
        temp[3][3] = 1.0f;
		
        return temp;
	}

	mat4f rotateZ(const float& angleInRads)
	{
		mat4f temp = identity_mat4f;
		temp[0][0] = cos(angleInRads);
		temp[0][1] = -sin(angleInRads);
		temp[1][0] = sin(angleInRads);
		temp[1][1] = cos(angleInRads);
        temp[3][3] = 1.0f;
		
        return temp;
	}

	mat4f rotate(const vec3f& axis, const float& angleInRads)
	{
		mat4f temp;
		const float sinA = sin(angleInRads), cosA = cos(angleInRads);
		temp[0][0] = cosA + ((1 - cosA) * powf(axis.x, 2));
		temp[0][1] = (axis.x * axis.y * (1 - cosA)) - (axis.z * sinA);
		temp[0][2] = (axis.x * axis.z * (1 - cosA)) + (axis.y * sinA);
		temp[1][0] = (axis.x * axis.y * (1 - cosA)) + (axis.z * sinA);
		temp[1][1] = cosA + ((1 - cosA) * powf(axis.y, 2));
		temp[1][2] = (axis.y * axis.z * (1 - cosA)) - (axis.x * sinA);
		temp[2][0] = (axis.x * axis.z * (1 - cosA)) - (axis.y * sinA);
		temp[2][1] = (axis.y * axis.z * (1 - cosA)) + (axis.x * sinA);
		temp[2][2] = cosA + ((1 - cosA) * powf(axis.z, 2));
		temp[3][3] = 1.0f;

		return temp;
	}

    vec3f rotation(const vec3f& vec, const vec3f& axis, float angle)
    {
        quatf func(cosf(angle/2.0f), axis * sinf(angle/2.0f));
        quatf out(0.0f, vec);
        out = (func * out) * func.Reciprocal();
        
        return out.VectorPart();
    }
	
    mat4f ortho(const float& xleft, const float& xright, const float& ybottom, const float& ytop, const float& znear, const float& zfar)
	{
		mat4f temp;

		temp[0][0] = 2.0f / (xright - xleft);
		temp[1][1] = 2.0f / (ytop - ybottom);
		temp[2][2] = -2.0f / (zfar - znear);
		temp[0][3] = -(xright + xleft) / (xright - xleft);
		temp[1][3] = -(ytop + ybottom) / (ytop - ybottom);
		temp[2][3] = -(zfar + znear) / (zfar - znear);
		temp[3][3] = 1.0f;

		return temp;
	}
	mat4f perspective(const float& FOVRads, const float& aspectRatio, const float& zNear, const float& zFar)
	{
		mat4f temp;
		temp[0][0] = 1.0f / (aspectRatio * tanf(FOVRads / 2.0f));
		temp[1][1] = 1.0f / tanf(FOVRads / 2.0f);
		temp[2][2] = zFar / (zFar - zNear);
        temp[3][2] = 1.0f;
        temp[2][3] = -zFar * zNear / (zFar - zNear);

		return temp;
	}

    mat4f lookAt(const vec3f& position, const vec3f& up, const vec3f& right)
    {
        vec3f direction = crossproduct(up, right).UnitVector();
        mat4f space = identity_mat4f;
        
        space[0][0] = right.x;
        space[0][1] = right.y;
        space[0][2] = right.z;
        space[1][0] = up.x;
        space[1][1] = up.y;
        space[1][2] = up.z;
        space[2][0] = direction.x;
        space[2][1] = direction.y;
        space[2][2] = direction.z;
        
        return multiply(space, translate(-position));
    }

mat4f lookAt(const vec3f& position, const vec3f& up, const vec3f& right, const vec3f& direction)
{
    mat4f space = identity_mat4f;
    
    space[0][0] = right.x;
    space[0][1] = right.y;
    space[0][2] = right.z;
    space[1][0] = up.x;
    space[1][1] = up.y;
    space[1][2] = up.z;
    space[2][0] = direction.x;
    space[2][1] = direction.y;
    space[2][2] = direction.z;
    
    return multiply(space, translate(-position));
}

    float angle(const vec3f& vec1, const vec3f& vec2)
    {
        return acos(dotproduct(vec1, vec2) / (vec1.Magnitude() * vec2.Magnitude()));
    }

}

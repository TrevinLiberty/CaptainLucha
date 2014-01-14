/****************************************************************************/
/* Copyright (c) 2013, Trevin Liberty
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
/****************************************************************************/
/*
 *	@author Trevin Liberty
 *	@file	
 *	@brief	
 *
/****************************************************************************/

#include "Matrix4D.h"

#define _USE_MATH_DEFINES
#include <math.h>

template<class Real>
Matrix4D<Real>::Matrix4D(bool isIdentity = false)
{
	if(isIdentity)
		*this = Matrix4D<Real>::IDENTITY;
	else
		*this = Matrix4D<Real>::ZERO;
}

template<class Real>
Matrix4D<Real>::Matrix4D(const Matrix4D<Real>& rhs)
{
	*this = rhs;
}

template<class Real>
Matrix4D<Real>::Matrix4D(Real radians, Real x, Real y, Real z)
{
	MakeRotationRad(radians, x, y, z);
}

template<class Real>
Matrix4D<Real>::Matrix4D(Real translateX, Real translateY, Real translateZ)
{
	*this = Matrix4D<Real>::IDENTITY;
	elements_[12] = translateX;
	elements_[13] = translateY;
	elements_[14] = translateZ;
}

template<class Real>
Matrix4D<Real>::Matrix4D(
	Real aa, Real ab, Real ac, Real ad,
	Real ba, Real bb, Real bc, Real bd,
	Real ca, Real cb, Real cc, Real cd,
	Real da, Real db, Real dc, Real dd)
{
	elements_[0] = aa;
	elements_[1] = ab;
	elements_[2] = ac;
	elements_[3] = ad;
	elements_[4] = ba;
	elements_[5] = bb;
	elements_[6] = bc;
	elements_[7] = bd;
	elements_[8] = ca;
	elements_[9] = cb;
	elements_[10] = cc;
	elements_[11] = cd;
	elements_[12] = da;
	elements_[13] = db;
	elements_[14] = dc;
	elements_[15] = dd;
}

template<class Real>
void Matrix4D<Real>::MakeZero()
{
	memset(elements_, 0, 16*sizeof(Real));
}

template<class Real>
void Matrix4D<Real>::MakeIdentity()
{
	elements_[0] = 1;
	elements_[1] = 0;
	elements_[2] = 0;
	elements_[3] = 0;
	elements_[4] = 0;
	elements_[5] = 1;
	elements_[6] = 0;
	elements_[7] = 0;
	elements_[8] = 0;
	elements_[9] = 0;
	elements_[10]= 1;
	elements_[11]= 0;
	elements_[12]= 0;
	elements_[13]= 0;
	elements_[14]= 0;
	elements_[15]= 1;
}

template<class Real>
Vector4D<Real> Matrix4D<Real>::GetRow(int i)
{
	return Vector4D<Real>(
		elements_[GetIndex(i, 0)], 
		elements_[GetIndex(i, 1)], 
		elements_[GetIndex(i, 2)],
		elements_[GetIndex(i, 3)]);
}

template<class Real>
Vector4D<Real> Matrix4D<Real>::GetColumn(int j)
{
	return Vector4D<Real>(
		elements_[GetIndex(0, j)], 
		elements_[GetIndex(1, j)],
		elements_[GetIndex(2, j)],
		elements_[GetIndex(3, j)]);
}

template<class Real>
Matrix3D<Real> Matrix4D<Real>::GetRotation() const
{
	return Matrix3D<Real>(
		elements_[0], elements_[1], elements_[2],
		elements_[4], elements_[5], elements_[6],
		elements_[8], elements_[9], elements_[10]);
}

template<class Real>
Vector3D<Real> Matrix4D<Real>::GetTranslate() const
{
	return Vector3D<Real>(elements_[12], elements_[13], elements_[14]);
}

template<class Real>
Vector3D<Real> Matrix4D<Real>::GetAxis(int i) const
{
	if(i == 0)
		return Vector3D<Real>(elements_[0], elements_[1], elements_[2]);
	if(i == 1)
		return Vector3D<Real>(elements_[4], elements_[5], elements_[6]);
	if(i == 2)
		return Vector3D<Real>(elements_[8], elements_[9], elements_[10]);
	if(i == 3)
		return Vector3D<Real>(elements_[12], elements_[13], elements_[14]);

	return Vector3D<Real>();

}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::MakeRotationRad(Real angle, Real x, Real y, Real z)
{
	float length = sqrt(x*x + y*y + z*z);
	x /= length;
	y /= length;
	z /= length;

	Real c = cos(angle);
	Real s = sin(angle);

	Real t = 1 - c;
	Real xx = x*x;
	Real yy = y*y;
	Real zz = z*z;
	Real xy = x*y;
	Real xz = x*z;
	Real yz = y*z;

	Real sx = s*x;
	Real sy = s*y;
	Real sz = s*z;

	elements_[0] = t*xx + c;
	elements_[1] = t*xy + sz;
	elements_[2] = t*xz - sy;
	elements_[3] = 0.0;

	elements_[4] = t*xy - sz;
	elements_[5] = t*yy + c;
	elements_[6] = t*yz + sx;
	elements_[7] = 0.0;

	elements_[8]  = t*xz + sy;
	elements_[9]  = t*yz - sx;
	elements_[10] = t*zz + c;
	elements_[11] = 0.0;

	elements_[12] = 0.0;
	elements_[13] = 0.0;
	elements_[14] = 0.0;
	elements_[15] = 1.0;

	return *this;
}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::MakeRotation(Real angle, Real x, Real y, Real z)
{
	MakeRotationRad((angle * (float)M_PI) / 180.0f, x, y, z);
	return *this;
}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::MakeRotation(const Vector3D<Real>& dir, const Vector3D<Real>& up)
{
	Vector3Df z(dir);
	z.Normalize();

	Vector3Df x = up.CrossProduct(z);
	x.Normalize();

	Vector3Df y = z.CrossProduct(x);

	*this = Matrix4Df
		(x.x, y.x, z.x, 0.0f,
		x.y, y.y, z.y, 0.0f,
		x.z, y.z, z.z, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	return *this;
}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::MakeTransformation(const Vector3D<Real>& trans, const Quaternion& orientation)
{
	elements_[0] = 1 - 2 * orientation.j * orientation.j - 2 * orientation.k * orientation.k;
	elements_[4] = 2 * orientation.i * orientation.j - 2 * orientation.r * orientation.k;
	elements_[8] = 2 * orientation.i * orientation.k + 2 * orientation.r * orientation.j;
	elements_[3] = 0.0f;
	elements_[1] = 2 * orientation.i * orientation.j + 2 * orientation.r * orientation.k;
	elements_[5] = 1 - 2 * orientation.i * orientation.i - 2 * orientation.k * orientation.k;
	elements_[9] = 2 * orientation.j * orientation.k - 2 * orientation.r * orientation.i;
	elements_[7] = 0.0f;
	elements_[2] = 2 * orientation.i * orientation.k - 2 * orientation.r * orientation.j;
	elements_[6] = 2 * orientation.j * orientation.k + 2 * orientation.r * orientation.i;
	elements_[10] = 1 - 2 * orientation.i * orientation.i - 2 * orientation.j * orientation.j;
	elements_[11] = 0.0f;

	elements_[12] = trans.x;
	elements_[13] = trans.y;
	elements_[14] = trans.z;
	elements_[15] = 1.0f;

	return *this;
}

template<class Real>
void Matrix4D<Real>::SetRotation(const Matrix3D<Real>& rotation)
{
	elements_[0] = rotation.GetValueAtIndex(0, 0);
	elements_[1] = rotation.GetValueAtIndex(1, 0);
	elements_[2] = rotation.GetValueAtIndex(2, 0);
	elements_[4] = rotation.GetValueAtIndex(0, 1);
	elements_[5] = rotation.GetValueAtIndex(1, 1);
	elements_[6] = rotation.GetValueAtIndex(2, 1);
	elements_[8] = rotation.GetValueAtIndex(0, 2);
	elements_[9] = rotation.GetValueAtIndex(1, 2);
	elements_[10] = rotation.GetValueAtIndex(2, 2);
	elements_[15] = 1.0;
}

template<class Real>
void Matrix4D<Real>::SetTranslate(const Vector3D<Real>& translate)
{
	elements_[12] = translate.x;
	elements_[13] = translate.y;
	elements_[14] = translate.z;
}

template<class Real>
void Matrix4D<Real>::SetSkewSymmetric(const Vector3D<Real>& vect)
{
	MakeIdentity();
	elements_[0] = elements_[5] = elements_[10] = 0;
	elements_[1] = -vect.z;
	elements_[2] = vect.y;
	elements_[4] = vect.z;
	elements_[6] = -vect.x;
	elements_[8] = -vect.y;
	elements_[9] = vect.x;
}

template<class Real>
Vector3D<Real> Matrix4D<Real>::TransformPosition(const Vector3D<Real>& vect) const
{
	Vector4D<Real> result = Transform(Vector4D<Real>(vect.x, vect.y, vect.z, 1.0f));
	return Vector3D<Real>(result.x, result.y, result.z);
}

template<class Real>
Vector3D<Real> Matrix4D<Real>::TransformRotation(const Vector3D<Real>& vect) const
{
	Vector4D<Real> result = Transform(Vector4D<Real>(vect.x, vect.y, vect.z, 0.0f));
	return Vector3D<Real>(result.x, result.y, result.z);
}

template<class Real>
Vector4D<Real> Matrix4D<Real>::Transform(const Vector4D<Real>& vect) const
{
	return *this * vect;
}

template<class Real>
Matrix4D<Real> Matrix4D<Real>::GetTranspose() const
{
	Matrix4D<Real> result;

	for(int row = 0; row < 4; ++row)
	{
		for(int col = 0; col < 4; ++col)
		{
			result.elements_[GetIndex(row, col)] = elements_[GetIndex(col, row)];
		}
	}
	return result;
}

template<class Real>
Real Matrix4D<Real>::GetDeterminant() const
{
	Real A0 = elements_[ 0]*elements_[ 5] - elements_[ 1]*elements_[ 4];
	Real A1 = elements_[ 0]*elements_[ 6] - elements_[ 2]*elements_[ 4];
	Real A2 = elements_[ 0]*elements_[ 7] - elements_[ 3]*elements_[ 4];
	Real A3 = elements_[ 1]*elements_[ 6] - elements_[ 2]*elements_[ 5];
	Real A4 = elements_[ 1]*elements_[ 7] - elements_[ 3]*elements_[ 5];
	Real A5 = elements_[ 2]*elements_[ 7] - elements_[ 3]*elements_[ 6];
	Real B0 = elements_[ 8]*elements_[13] - elements_[ 9]*elements_[12];
	Real B1 = elements_[ 8]*elements_[14] - elements_[10]*elements_[12];
	Real B2 = elements_[ 8]*elements_[15] - elements_[11]*elements_[12];
	Real B3 = elements_[ 9]*elements_[14] - elements_[10]*elements_[13];
	Real B4 = elements_[ 9]*elements_[15] - elements_[11]*elements_[13];
	Real B5 = elements_[10]*elements_[15] - elements_[11]*elements_[14];

	return A0*B5 - A1*B4 + A2*B3 + A3*B2 - A4*B1 + A5*B0;
}

template<class Real>
Matrix4D<Real> Matrix4D<Real>::GetInverse() const
{
	const Real* m = elements_;
	Matrix4D<Real> inv;
	Real det;
	int i;

	inv[0] = m[5]  * m[10] * m[15] - 
		m[5]  * m[11] * m[14] - 
		m[9]  * m[6]  * m[15] + 
		m[9]  * m[7]  * m[14] +
		m[13] * m[6]  * m[11] - 
		m[13] * m[7]  * m[10];

	inv[4] = -m[4]  * m[10] * m[15] + 
		m[4]  * m[11] * m[14] + 
		m[8]  * m[6]  * m[15] - 
		m[8]  * m[7]  * m[14] - 
		m[12] * m[6]  * m[11] + 
		m[12] * m[7]  * m[10];

	inv[8] = m[4]  * m[9] * m[15] - 
		m[4]  * m[11] * m[13] - 
		m[8]  * m[5] * m[15] + 
		m[8]  * m[7] * m[13] + 
		m[12] * m[5] * m[11] - 
		m[12] * m[7] * m[9];

	inv[12] = -m[4]  * m[9] * m[14] + 
		m[4]  * m[10] * m[13] +
		m[8]  * m[5] * m[14] - 
		m[8]  * m[6] * m[13] - 
		m[12] * m[5] * m[10] + 
		m[12] * m[6] * m[9];

	inv[1] = -m[1]  * m[10] * m[15] + 
		m[1]  * m[11] * m[14] + 
		m[9]  * m[2] * m[15] - 
		m[9]  * m[3] * m[14] - 
		m[13] * m[2] * m[11] + 
		m[13] * m[3] * m[10];

	inv[5] = m[0]  * m[10] * m[15] - 
		m[0]  * m[11] * m[14] - 
		m[8]  * m[2] * m[15] + 
		m[8]  * m[3] * m[14] + 
		m[12] * m[2] * m[11] - 
		m[12] * m[3] * m[10];

	inv[9] = -m[0]  * m[9] * m[15] + 
		m[0]  * m[11] * m[13] + 
		m[8]  * m[1] * m[15] - 
		m[8]  * m[3] * m[13] - 
		m[12] * m[1] * m[11] + 
		m[12] * m[3] * m[9];

	inv[13] = m[0]  * m[9] * m[14] - 
		m[0]  * m[10] * m[13] - 
		m[8]  * m[1] * m[14] + 
		m[8]  * m[2] * m[13] + 
		m[12] * m[1] * m[10] - 
		m[12] * m[2] * m[9];

	inv[2] = m[1]  * m[6] * m[15] - 
		m[1]  * m[7] * m[14] - 
		m[5]  * m[2] * m[15] + 
		m[5]  * m[3] * m[14] + 
		m[13] * m[2] * m[7] - 
		m[13] * m[3] * m[6];

	inv[6] = -m[0]  * m[6] * m[15] + 
		m[0]  * m[7] * m[14] + 
		m[4]  * m[2] * m[15] - 
		m[4]  * m[3] * m[14] - 
		m[12] * m[2] * m[7] + 
		m[12] * m[3] * m[6];

	inv[10] = m[0]  * m[5] * m[15] - 
		m[0]  * m[7] * m[13] - 
		m[4]  * m[1] * m[15] + 
		m[4]  * m[3] * m[13] + 
		m[12] * m[1] * m[7] - 
		m[12] * m[3] * m[5];

	inv[14] = -m[0]  * m[5] * m[14] + 
		m[0]  * m[6] * m[13] + 
		m[4]  * m[1] * m[14] - 
		m[4]  * m[2] * m[13] - 
		m[12] * m[1] * m[6] + 
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] + 
		m[1] * m[7] * m[10] + 
		m[5] * m[2] * m[11] - 
		m[5] * m[3] * m[10] - 
		m[9] * m[2] * m[7] + 
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] - 
		m[0] * m[7] * m[10] - 
		m[4] * m[2] * m[11] + 
		m[4] * m[3] * m[10] + 
		m[8] * m[2] * m[7] - 
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] + 
		m[0] * m[7] * m[9] + 
		m[4] * m[1] * m[11] - 
		m[4] * m[3] * m[9] - 
		m[8] * m[1] * m[7] + 
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] - 
		m[0] * m[6] * m[9] - 
		m[4] * m[1] * m[10] + 
		m[4] * m[2] * m[9] + 
		m[8] * m[1] * m[6] - 
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (FloatEquality(det, 0))
		return Matrix4D<Real>::IDENTITY;

	det = 1.0f / det;

	Matrix4D<Real> reuslt;
	for (i = 0; i < 16; i++)
		reuslt[i] = inv[i] * det;

	return reuslt;
}

template<class Real>
void Matrix4D<Real>::CreateLookAt(const Vector3Df& pos, const Vector3Df& lookAt, const Vector3Df& up)
{
	Vector3Df z = (pos - lookAt);
	z.Normalize();

	Vector3Df x = up.CrossProduct(z);
	x.Normalize();

	Vector3Df y = z.CrossProduct(x);

	*this = Matrix4Df
		(x.x, y.x, z.x, 0.0f,
		 x.y, y.y, z.y, 0.0f,
		 x.z, y.z, z.z, 0.0f,
		 -x.Dot(pos), -y.Dot(pos), -z.Dot(pos), 1.0f);
}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::operator= (const Matrix4D<Real>& rhs)
{
	memcpy(elements_, rhs.elements_, 16*sizeof(Real));
	return *this;
}

template<class Real>
bool Matrix4D<Real>::operator== (const Matrix4D<Real>& rhs) const
{
	return CompareArrays(rhs) == 0;
}

template<class Real>
bool Matrix4D<Real>::operator!= (const Matrix4D<Real>& rhs) const
{
	return CompareArrays(rhs) != 0;
}

template<class Real>
Matrix4D<Real> Matrix4D<Real>::operator+ (const Matrix4D<Real>& rhs) const
{
	Matrix4D<Real> result = *this;
	for (int i = 0; i < 16; i++)
		result.elements_[i] += rhs.elements_[i];
	return result;
}

template<class Real>
Matrix4D<Real> Matrix4D<Real>::operator- (const Matrix4D<Real>& rhs) const
{
	Matrix4D<Real> result = *this;
	for (int i = 0; i < 16; i++)
		result.elements_[i] -= rhs.elements_[i];
	return result;
}

template<class Real>
Matrix4D<Real> Matrix4D<Real>::operator* (const Matrix4D<Real>& rhs) const
{
	Matrix4D<Real> result;
	for(int row = 0; row < 4; ++row)
	{
		for(int col = 0; col < 4; ++col)
		{
			int i = GetIndex(col, row); /*col * row * 4*/
			for(int j = 0; j < 4; ++j)
				result.elements_[i] += elements_[GetIndex(j, row)] * rhs.elements_[GetIndex(col, j)];
		}
	}

	return result;
}

template<class Real>
Matrix4D<Real> Matrix4D<Real>::operator* (Real scalar) const
{
	Matrix4D<Real> result = *this;
	for (int i = 0; i < 16; i++)
		result.elements_[i] *= scalar;
	return result;
}

template<class Real>
Matrix4D<Real> Matrix4D<Real>::operator/ (Real scalar) const
{
	Matrix4D<Real> result = *this;
	for (int i = 0; i < 16; i++)
		result.elements_[i] /= scalar;
	return result;
}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::operator+= (const Matrix4D<Real>& rhs)
{
	for (int i = 0; i < 16; i++)
		elements_[i] += rhs.elements_[i];
	return *this;
}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::operator-= (const Matrix4D<Real>& rhs)
{
	for (int i = 0; i < 16; i++)
		elements_[i] -= rhs.elements_[i];
	return *this;
}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::operator*= (Real scalar)
{
	for (int i = 0; i < 16; i++)
		elements_[i] *= scalar;
	return *this;
}

template<class Real>
Matrix4D<Real>& Matrix4D<Real>::operator/= (Real scalar)
{
	for (int i = 0; i < 16; i++)
		elements_[i] /= scalar;
	return *this;
}

template<class Real>
Vector4D<Real> Matrix4D<Real>::operator* (const Vector4D<Real>& rhs) const
{
	return Vector4D<Real>(
		rhs.x*elements_[0] + rhs.y*elements_[4] + rhs.z*elements_[8] + rhs.w*elements_[12],
		rhs.x*elements_[1] + rhs.y*elements_[5] + rhs.z*elements_[9] + rhs.w*elements_[13],
		rhs.x*elements_[2] + rhs.y*elements_[6] + rhs.z*elements_[10] + rhs.w*elements_[14],
		rhs.x*elements_[3] + rhs.y*elements_[7] + rhs.z*elements_[11] + rhs.w*elements_[15]);
}

template<class Real>
Matrix4D<Real>::operator Matrix4D<float>() const
{
	Matrix4D<float> result;
	result[0] = static_cast<float>(elements_[0] );
	result[1] = static_cast<float>(elements_[1] );
	result[2] = static_cast<float>(elements_[2] );
	result[3] = static_cast<float>(elements_[3] );
	result[4] = static_cast<float>(elements_[4] );
	result[5] = static_cast<float>(elements_[5] );
	result[6] = static_cast<float>(elements_[6] );
	result[7] = static_cast<float>(elements_[7] );
	result[8] = static_cast<float>(elements_[8] );
	result[9] = static_cast<float>(elements_[9] );
	result[10]= static_cast<float>(elements_[10]);
	result[11]= static_cast<float>(elements_[11]);
	result[12]= static_cast<float>(elements_[12]);
	result[13]= static_cast<float>(elements_[13]);
	result[14]= static_cast<float>(elements_[14]);
	result[15]= static_cast<float>(elements_[15]);
	return result;
}

template<class Real>
Matrix4D<Real>::operator Matrix4D<double>() const
{
	Matrix4D<double> result;
	result[0] = static_cast<double>(elements_[0] );
	result[1] = static_cast<double>(elements_[1] );
	result[2] = static_cast<double>(elements_[2] );
	result[3] = static_cast<double>(elements_[3] );
	result[4] = static_cast<double>(elements_[4] );
	result[5] = static_cast<double>(elements_[5] );
	result[6] = static_cast<double>(elements_[6] );
	result[7] = static_cast<double>(elements_[7] );
	result[8] = static_cast<double>(elements_[8] );
	result[9] = static_cast<double>(elements_[9] );
	result[10]= static_cast<double>(elements_[10]);
	result[11]= static_cast<double>(elements_[11]);
	result[12]= static_cast<double>(elements_[12]);
	result[13]= static_cast<double>(elements_[13]);
	result[14]= static_cast<double>(elements_[14]);
	result[15]= static_cast<double>(elements_[15]);
	return result;
}

//////////////////////////////////////////////////////////////////////////
//	Protected
//////////////////////////////////////////////////////////////////////////
template<class Real>
int Matrix4D<Real>::GetIndex(int i, int j) const
{
	return i + j * 4;
}

template<class Real>
int Matrix4D<Real>::CompareArrays (const Matrix4D<Real>& rhs) const
{
	return memcmp(elements_, rhs.elements_, 16*sizeof(Real));
}
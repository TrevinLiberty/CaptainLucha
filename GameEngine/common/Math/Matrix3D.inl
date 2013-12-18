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

#include "Matrix3D.h"

template<class Real>
Matrix3D<Real>::Matrix3D(bool isIdentity = false)
{
	if(isIdentity)
		MakeIdentity();
	else
		MakeZero();
}

template<class Real>
Matrix3D<Real>::Matrix3D(const Matrix3D<Real>& rhs)
{
	*this = rhs;
}

template<class Real>
Matrix3D<Real>::Matrix3D(
	Real aa, Real ab, Real ac, 
	Real ba, Real bb, Real bc,
	Real ca, Real cb, Real cc)
{
	elements_[0] = aa;
	elements_[1] = ab;
	elements_[2] = ac;
	elements_[3] = ba;
	elements_[4] = bb;
	elements_[5] = bc;
	elements_[6] = ca;
	elements_[7] = cb;
	elements_[8] = cc;
}

template<class Real>
Matrix3D<Real>::Matrix3D(
	const Vector3D<Real>& r1, 
	const Vector3D<Real>& r2, 
	const Vector3D<Real>& r3)
{
	elements_[0] = r1.x;
	elements_[3] = r1.y;
	elements_[6] = r1.z;

	elements_[1] = r2.x;
	elements_[4] = r2.y;
	elements_[7] = r2.z;

	elements_[2] = r3.x;
	elements_[5] = r3.y;
	elements_[8] = r3.z;
}

template<class Real>
void Matrix3D<Real>::MakeZero()
{
	memset(elements_, 0, 9*sizeof(Real));
}

template<class Real>
void Matrix3D<Real>::MakeIdentity()
{
	elements_[0] = 1;
	elements_[1] = 0;
	elements_[2] = 0;
	elements_[3] = 0;
	elements_[4] = 1;
	elements_[5] = 0;
	elements_[6] = 0;
	elements_[7] = 0;
	elements_[8] = 1;
}

template<class Real>
void Matrix3D<Real>::MakeRotation(const Vector3D<Real>& axisIN, Real angle)
{
	Vector3D<Real> axis = axisIN;
	axis.Normalize();

	Real c = cos(angle);
	Real s = sin(angle);

	Real t = 1 - c;
	Real xx = axis.x*axis.x;
	Real yy = axis.y*axis.y;
	Real zz = axis.z*axis.z;
	Real xy = axis.x*axis.y;
	Real xz = axis.x*axis.z;
	Real yz = axis.y*axis.z;

	Real sx = s*axis.x;
	Real sy = s*axis.y;
	Real sz = s*axis.z;

	Matrix3D<Real> result(
		t*xx + c, t*xy - sz, t*xz + sy,    
		t*xy + sz, t*yy + c, t*yz - sx,    
		t*xz - sy, t*yz + sx, t*zz + c);
	(*this) = result.GetTranspose();
}

template<class Real>
void Matrix3D<Real>::MakeRotation(Real pitch, Real yaw, Real roll = 0)
{
	Real cx = cos(pitch);
	Real sx = sin(pitch);
	Real cy = cos(yaw);
	Real sy = sin(yaw);
	Real cz = cos(roll);
	Real sz = sin(roll);

	elements_[0] = cy * cz;
	elements_[1] = cy * sz;
	elements_[2] = -sy;
	elements_[3] = -cx * sz + sx * sy * cz;
	elements_[4] = cx * cz + sx * sy * sz;
	elements_[5] = sx * cy;
	elements_[6] = sx * sz + cx * sy * cz;
	elements_[7] = -sx * cz + cx * sy * sz;
	elements_[8] = cx * cy;
}

template<class Real>
Vector3D<Real> Matrix3D<Real>::GetAxis(int i) const
{
	if(i == 0)
		return Vector3D<Real>(elements_[0], elements_[1], elements_[2]);
	if(i == 1)
		return Vector3D<Real>(elements_[3], elements_[4], elements_[5]);
	if(i == 2)
		return Vector3D<Real>(elements_[6], elements_[7], elements_[8]);

	return Vector3D<Real>();
}

template<class Real>
void Matrix3D<Real>::SetSkewSymmetric(const Vector3D<Real>& vect)
{
	MakeIdentity();
	elements_[0] = elements_[4] = elements_[8] = 0;
	elements_[1] = -vect.z;
	elements_[2] = vect.y;
	elements_[3] = vect.z;
	elements_[5] = -vect.x;
	elements_[6] = -vect.y;
	elements_[7] = vect.x;
	*this = GetTranspose();
}

template<class Real>
Real Matrix3D<Real>::GetValueAtIndex(int i, int j) const
{
	return elements_[GetIndex(i, j)];
}

template<class Real>
Matrix3D<Real> Matrix3D<Real>::GetTranspose() const
{
	Matrix3D<Real> result = *this;

	result.elements_[3] = elements_[1];
	result.elements_[1] = elements_[3];

	result.elements_[6] = elements_[2];
	result.elements_[2] = elements_[6];

	result.elements_[7] = elements_[5];
	result.elements_[5] = elements_[7];

	return result;
}

template<class Real>
Matrix3D<Real> Matrix3D<Real>::GetInverse() const
{
	Matrix3D<Real> result;
	
	result.elements_[0] = elements_[4]*elements_[8] - elements_[5]*elements_[7];
	result.elements_[3] = elements_[2]*elements_[7] - elements_[1]*elements_[8];
	result.elements_[6] = elements_[1]*elements_[5] - elements_[2]*elements_[4];
	result.elements_[1] = elements_[5]*elements_[6] - elements_[3]*elements_[8];
	result.elements_[4] = elements_[0]*elements_[8] - elements_[2]*elements_[6];
	result.elements_[7] = elements_[2]*elements_[3] - elements_[0]*elements_[5];
	result.elements_[2] = elements_[3]*elements_[7] - elements_[4]*elements_[6];
	result.elements_[5] = elements_[1]*elements_[6] - elements_[0]*elements_[7];
	result.elements_[8] = elements_[0]*elements_[4] - elements_[1]*elements_[3];

	Real t = 
		elements_[0]*result.elements_[GetIndex(0, 0)]
	  + elements_[1]*result.elements_[GetIndex(1, 0)] 
	  + elements_[2]*result.elements_[GetIndex(2, 0)];

	if(abs(t) < 0.00000001)
	{
		//return result.MakeZero();
	}

	result /= t;
	return result.GetTranspose();
}

template<class Real>
Matrix3D<Real>& Matrix3D<Real>::operator= (const Matrix3D<Real>& rhs)
{
	memcpy(elements_, rhs.elements_, 9*sizeof(Real));
	return *this;
}

template<class Real>
bool Matrix3D<Real>::operator== (const Matrix3D<Real>& rhs) const
{
	return CompareArrays(rhs) == 0;
}

template<class Real>
bool Matrix3D<Real>::operator!= (const Matrix3D<Real>& rhs) const
{
	return CompareArrays(rhs) != 0;
}

template<class Real>
Matrix3D<Real> Matrix3D<Real>::operator+ (const Matrix3D<Real>& rhs) const
{
	Matrix3D<Real> result = *this;
	for (int i = 0; i < 9; i++)
		result.elements_[i] += rhs.elements_[i];
	return result;
}

template<class Real>
Matrix3D<Real> Matrix3D<Real>::operator- (const Matrix3D<Real>& rhs) const
{
	Matrix3D<Real> result = *this;
	for (int i = 0; i < 9; i++)
		result.elements_[i] -= rhs.elements_[i];
	return result;
}

template<class Real>
Matrix3D<Real> Matrix3D<Real>::operator* (const Matrix3D<Real>& rhs) const
{
	Matrix3D<Real> result;
	for(int row = 0; row < 3; ++row)
	{
		for(int col = 0; col < 3; ++col)
		{
			int i = GetIndex(col, row);
			for(int j = 0; j < 3; ++j)
				result.elements_[i] += elements_[GetIndex(j, row)] * rhs.elements_[GetIndex(col, j)];
		}
	}

	return result;
}

template<class Real>
Matrix3D<Real> Matrix3D<Real>::operator* (Real scalar) const
{
	Matrix3D<Real> result = *this;
	for (int i = 0; i < 9; i++)
		result.elements_[i] *= scalar;
	return result;
}

template<class Real>
Matrix3D<Real> Matrix3D<Real>::operator/ (Real scalar) const
{
	Matrix3D<Real> result = *this;
	for (int i = 0; i < 9; i++)
		result.elements_[i] /= scalar;
	return result;
}

template<class Real>
Matrix3D<Real>& Matrix3D<Real>::operator+= (const Matrix3D<Real>& rhs)
{
	for (int i = 0; i < 9; i++)
		elements_[i] += rhs.elements_[i];
	return *this;
}

template<class Real>
Matrix3D<Real>& Matrix3D<Real>::operator-= (const Matrix3D<Real>& rhs)
{
	for (int i = 0; i < 9; i++)
		elements_[i] -= rhs.elements_[i];
	return *this;
}

template<class Real>
Matrix3D<Real>& Matrix3D<Real>::operator*= (Real scalar)
{
	for (int i = 0; i < 9; i++)
		elements_[i] *= scalar;
	return *this;
}

template<class Real>
Matrix3D<Real>& Matrix3D<Real>::operator/= (Real scalar)
{
	for (int i = 0; i < 9; i++)
		elements_[i] /= scalar;
	return *this;
}

template<class Real>
Vector3D<Real> Matrix3D<Real>::operator* (const Vector3D<Real>& rhs) const
{
	return Vector3D<Real>(
		rhs.x*elements_[0] + rhs.y*elements_[3] + rhs.z*elements_[6],
		rhs.x*elements_[1] + rhs.y*elements_[4] + rhs.z*elements_[7],
		rhs.x*elements_[2] + rhs.y*elements_[5] + rhs.z*elements_[8]);
}

template<class Real>
Matrix3D<Real>::operator Matrix3D<float>() const
{
	Matrix3D<float> result;
	result[0] = static_cast<float>(elements_[0]);
	result[1] = static_cast<float>(elements_[1]);
	result[2] = static_cast<float>(elements_[2]);
	result[3] = static_cast<float>(elements_[3]);
	result[4] = static_cast<float>(elements_[4]);
	result[5] = static_cast<float>(elements_[5]);
	result[6] = static_cast<float>(elements_[6]);
	result[7] = static_cast<float>(elements_[7]);
	result[8] = static_cast<float>(elements_[8]);
	return result;
}

template<class Real>
Matrix3D<Real>::operator Matrix3D<double>() const
{
	Matrix3D<double> result;
	result[0] = static_cast<double>(elements_[0]);
	result[1] = static_cast<double>(elements_[1]);
	result[2] = static_cast<double>(elements_[2]);
	result[3] = static_cast<double>(elements_[3]);
	result[4] = static_cast<double>(elements_[4]);
	result[5] = static_cast<double>(elements_[5]);
	result[6] = static_cast<double>(elements_[6]);
	result[7] = static_cast<double>(elements_[7]);
	result[8] = static_cast<double>(elements_[8]);
	return result;
}

//////////////////////////////////////////////////////////////////////////
//	Protected
//////////////////////////////////////////////////////////////////////////
template<class Real>
int Matrix3D<Real>::GetIndex(int i, int j) const
{
	return i + j * 3;
}

template<class Real>
int Matrix3D<Real>::CompareArrays (const Matrix3D<Real>& rhs) const
{
	return memcmp(elements_, rhs.elements_, 9*sizeof(Real));
}
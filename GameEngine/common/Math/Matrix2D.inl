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

#include "Matrix2D.h"

template<class Real>
Matrix2D<Real>::Matrix2D(bool isIdentity = false)
{
	if(isIdentity)
		*this = Matrix2D<Real>::IDENTITY;
	else
		*this = Matrix2D<Real>::ZERO;
}

template<class Real>
Matrix2D<Real>::Matrix2D(const Matrix2D<Real>& rhs)
{
	*this = rhs;
}

template<class Real>
Matrix2D<Real>::Matrix2D(
	Real aa, Real ab,
	Real ba, Real bb)
{
	elements_[0] = aa;
	elements_[1] = ab;
	elements_[2] = ba;
	elements_[3] = bb;
}

template<class Real>
Matrix2D<Real>::Matrix2D(const Vector2D<Real>& r1, const Vector2D<Real>& r2)
{
	elements_[0] = r1.x_;
	elements_[1] = r1.y_;
	elements_[2] = r2.x_;
	elements_[3] = r2.y_;
}

template<class Real>
void Matrix2D<Real>::MakeZero()
{
	memset(m_afEntry,0,4*sizeof(Real));
}

template<class Real>
void Matrix2D<Real>::MakeIdentity()
{
	*this = Matrix2D<Real>::IDENTITY;
}

template<class Real>
Vector2D<Real> Matrix2D<Real>::GetRow(int i)
{
	return Vector2D<Real>(
		elements_[GetIndex(i, 0)], 
		elements_[GetIndex(i, 1)]);
}
	
template<class Real>
Vector2D<Real> Matrix2D<Real>::GetColumn(int j)
{
	return Vector2D<Real>(
		elements_[GetIndex(0, j)], 
		elements_[GetIndex(1, j)]);
}

template<class Real>
void Matrix2D<Real>::SetRow(int row, const Vector2D<Real>& val)
{
	elements_[GetIndex(row, 0)] = val.x_;
	elements_[GetIndex(row, 1)] = val.y_;
}
	
template<class Real>
void Matrix2D<Real>::SetColumn(int col, const Vector2D<Real>& val)
{
	elements_[GetIndex(0, col)] = val.x_;
	elements_[GetIndex(1, col)] = val.y_;
}

template<class Real>
Real Matrix2D<Real>::GetValueAtIndex(int i, int j) const
{
	return elements_[GetIndex(i, j)];
}

template<class Real>
Matrix2D<Real> Matrix2D<Real>::GetTranspose() const
{
	Matrix2D<Real> result = *this;

	Real t = result.elements_[2];
	result.elements_[2] = result.elements_[1];
	result.elements_[1] = t;

	return result;
}
	
template<class Real>
Matrix2D<Real> Matrix2D<Real>::GetInverse() const
{
	Matrix2D<Real> result;
	Real t = elements_[0] * elements_[3] - elements_[1] * elements_[2];

	if(abs(t) > 0.00000001)
	{
		t = 1 / t;
		result.elements_[0] = elements_[3] * t;
		result.elements_[1] = -elements_[1] * t;
		result.elements_[2] = -elements_[2] * t;
		result.elements_[3] = elements_[0] * t;
	}
	else
		return ZERO;

	return result;
}

template<class Real>
Matrix2D<Real>& Matrix2D<Real>::operator= (const Matrix2D<Real>& rhs)
{
	memcpy(elements_, rhs.elements_, 4*sizeof(Real));
	return *this;
}

template<class Real>
bool Matrix2D<Real>::operator== (const Matrix2D<Real>& rhs) const
{
	return CompareArrays(rhs) == 0;
}
	
template<class Real>
bool Matrix2D<Real>::operator!= (const Matrix2D<Real>& rhs) const
{
	return CompareArrays(rhs) != 0;
}

template<class Real>
Matrix2D<Real> Matrix2D<Real>::operator+ (const Matrix2D<Real>& rhs) const
{
	Matrix2D<Real> result = *this;
	for (int i = 0; i < 4; i++)
		result.elements_[i] += rhs.elements_[i];
	return result;
}
	
template<class Real>
Matrix2D<Real> Matrix2D<Real>::operator- (const Matrix2D<Real>& rhs) const
{
	Matrix2D<Real> result = *this;
	for (int i = 0; i < 4; i++)
		result.elements_[i] -= rhs.elements_[i];
	return result;
}
	
template<class Real>
Matrix2D<Real> Matrix2D<Real>::operator* (const Matrix2D<Real>& rhs) const
{
	Matrix2D<Real> result;
	for(int row = 0; row < 2; ++row)
	{
		for(int col = 0; col < 2; ++col)
		{
			int i = GetIndex(col, row);
			for(int j = 0; j < 2; ++j)
				result.elements_[i] += elements_[GetIndex(j, row)] * rhs.elements_[GetIndex(col, j)];
		}
	}

	return result;
}
	
template<class Real>
Matrix2D<Real> Matrix2D<Real>::operator* (Real scalar) const
{
	Matrix2D<Real> result = *this;
	for (int i = 0; i < 4; i++)
		result.elements_[i] *= scalar;
	return result;
}
	
template<class Real>
Matrix2D<Real> Matrix2D<Real>::operator/ (Real scalar) const
{
	Matrix2D<Real> result = *this;
	for (int i = 0; i < 4; i++)
		result.elements_[i] /= scalar;
	return result;
}

template<class Real>
Matrix2D<Real>& Matrix2D<Real>::operator+= (const Matrix2D<Real>& rhs)
{
	for (int i = 0; i < 4; i++)
		elements_[i] += rhs.elements_[i];
	return *this;
}

template<class Real>
Matrix2D<Real>& Matrix2D<Real>::operator-= (const Matrix2D<Real>& rhs)
{
	for (int i = 0; i < 4; i++)
		elements_[i] -= rhs.elements_[i];
	return *this;
}

template<class Real>
Matrix2D<Real>& Matrix2D<Real>::operator*= (Real scalar)
{
	for (int i = 0; i < 4; i++)
		elements_[i] *= scalar;
	return *this;
}

template<class Real>
Matrix2D<Real>& Matrix2D<Real>::operator/= (Real scalar)
{
	for (int i = 0; i < 4; i++)
		elements_[i] /= scalar;
	return *this;
}

//////////////////////////////////////////////////////////////////////////
//	Protected
//////////////////////////////////////////////////////////////////////////
template<class Real>
int Matrix2D<Real>::GetIndex(int i, int j) const
{
	return i + j * 2;
}

template<class Real>
int Matrix2D<Real>::CompareArrays (const Matrix2D<Real>& rhs) const
{
	return memcmp(elements_, rhs.elements_, 4*sizeof(Real));
}
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

#include "Quaternion.h"
#include "Math.h"

namespace CaptainLucha
{
	Quaternion::Quaternion(Vector3D<Real> axis, Real angle)
	{
		r = cos(angle/2.0f);
		Real sinHalf = sin(angle/2.0f);

		i = axis.x * sinHalf;
		j = axis.y * sinHalf;
		k = axis.z * sinHalf;
	}

	Quaternion::Quaternion(const Quaternion& rhs)
	{
		this->r = rhs.r;
		this->i = rhs.i;
		this->j = rhs.j;
		this->k = rhs.k;
	}

	Quaternion::Quaternion(Real r, Real i, Real j, Real k)
	{
		this->r = r;
		this->i = i;
		this->j = j;
		this->k = k;
	}

	Quaternion::Quaternion()
	{
		this->r = 1.0f;
		this->i = 0.0f;
		this->j = 0.0f;
		this->k = 0.0f;
	}

	Quaternion::~Quaternion()
	{

	}

	void Quaternion::Normalize()
	{
		Real t = r*r + i*i + j*j + k*k;

		if(!FloatEquality(t, 0.0))
		{
			t = (Real)1.0 / std::sqrt(t);
			r *= t;
			i *= t;
			j *= t;
			k *= t;
		}
		else
			r = 1.0f;
	}

	Matrix3D<Real> Quaternion::GetMatrix() const
	{
		Real ii = i*i;
		Real jj = j*j;
		Real kk = k*k;

		Real ij = i*j;
		Real ik = i*k;
		Real jk = j*k;
		
		Real ri = r*i;
		Real rj = r*j;
		Real rk = r*k;

		return Matrix3D<Real>(
			1 - (2*jj + 2*kk), 2*ij + 2*rk, 2*ik - 2*rj,
			2*ij - 2*rk, 1 - (2*ii + 2*kk), 2*jk + 2*ri,
			2*ik + 2*rj, 2*jk - 2*ri, 1 - (2*ii + 2*jj));
	}

	Quaternion Quaternion::operator*(const Quaternion& rhs) const
	{
		Quaternion result(*this);
		result *= rhs;
		return result;
	}

	void Quaternion::operator*=(const Quaternion& rhs)
	{
		Quaternion q(*this);

		r = q.r*rhs.r - q.i*rhs.i -
			q.j*rhs.j - q.k*rhs.k;
		i = q.r*rhs.i + q.i*rhs.r +
			q.j*rhs.k - q.k*rhs.j;
		j = q.r*rhs.j + q.j*rhs.r +
			q.k*rhs.i - q.i*rhs.k;
		k = q.r*rhs.k + q.k*rhs.r +
			q.i*rhs.j - q.j*rhs.i;
	}

	void Quaternion::Rotate(const Vector3D<Real>& vector)
	{
		Quaternion q((Real)0.0, vector.x, vector.y, vector.z);
		*this *= q;
	}

	void Quaternion::AddScaledVector(const Vector3D<Real>& vector, Real scale)
	{
		Quaternion q((Real)0.0, vector.x * scale, vector.y * scale, vector.z * scale);
		q *= *this;

		r += q.r * ((Real)0.5);
		i += q.i * ((Real)0.5);
		j += q.j * ((Real)0.5);
		k += q.k * ((Real)0.5);
	}
}
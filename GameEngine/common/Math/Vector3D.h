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
 *	@file	Vector3D.h
 *	@brief	
 *
/****************************************************************************/

#ifndef VECTOR_3D_H_CL
#define VECTOR_3D_H_CL

#include <cmath>
#include <iostream>

namespace CaptainLucha
{
	template<class Real>
	class Vector3D
	{
	public:
		Vector3D() : x((Real)0.0), y((Real)0.0), z((Real)0.0) {};
		Vector3D(Real x, Real y, Real z) : x(x), y(y), z(z) {};
		Vector3D(const Vector3D<Real>& vect) : x(vect.x), y(vect.y), z(vect.z) {};

		Real x;
		Real y;
		Real z;

		Vector3D& operator= (const Vector3D& rhs);

		template<typename OtherT>
		operator Vector3D<OtherT>() 
		{
			return Vector3D<OtherT>((OtherT)x, (OtherT)y, (OtherT)z);
		}

		bool operator==(const Vector3D& rhs) const;
		bool operator!=(const Vector3D& rhs) const;

		Vector3D operator+(const Vector3D& rhs) const;
		Vector3D operator-(const Vector3D& rhs) const;
		Vector3D operator*(Real scalar) const;
		Vector3D operator/(Real scalar) const;

		Vector3D& operator+=(const Vector3D& rkV);
		Vector3D& operator-=(const Vector3D& rkV);
		Vector3D& operator*=(Real fScalar);
		Vector3D& operator/=(Real fScalar);

		Real& operator[] (int i);
		Real operator[] (int i) const;

		template<typename J>
		friend std::ostream& operator<<(std::ostream& out, const Vector3D<J>& vect);

		operator Vector3D<int>() const;
		operator Vector3D<float>() const;
		operator Vector3D<double>() const;

		Real Length() const;
		Real SquaredLength() const;
		Real Dot(const Vector3D& rhs) const;

		void Normalize();

		Vector3D CrossProduct(const Vector3D& rhs) const;

		bool Equals(const Vector3D& rhs) const;
	};

	template<typename J>
	std::ostream& operator<<(std::ostream& out, const Vector3D<J>& vect)
	{
		out << vect.x << " " << vect.y << " " << vect.z;
		return out;
	}

	typedef Vector3D<int> Vector3Di;
	typedef Vector3D<float> Vector3Df;
	typedef Vector3D<double> Vector3Dd;

	#include "Vector3D.inl"
}

#endif
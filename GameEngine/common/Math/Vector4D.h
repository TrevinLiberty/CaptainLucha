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
 *	@file	Vector4D.h
 *	@brief	
 *
/****************************************************************************/

#ifndef VECTOR_4D_H_CL
#define VECTOR_4D_H_CL

#include "Vector3D.h"

namespace CaptainLucha
{
	template<class Real>
	class Vector4D
	{
	public:
		Vector4D() : x((Real)0.0), y((Real)0.0), z((Real)0.0), w((Real)0.0) {};
		Vector4D(Real x, Real y, Real z, Real w) : x(x), y(y), z(z), w(w) {};
		Vector4D(const Vector4D<Real>& vect) : x(vect.x), y(vect.y), z(vect.z), w(vect.w) {};

		Real x;
		Real y;
		Real z;
		Real w;

		Vector4D& operator= (const Vector4D& rhs);

		bool operator==(const Vector4D& rhs) const;
		bool operator!=(const Vector4D& rhs) const;

		Vector4D operator+(const Vector4D& rhs) const;
		Vector4D operator-(const Vector4D& rhs) const;
		Vector4D operator*(Real scalar) const;
		Vector4D operator/(Real scalar) const;

		Vector4D& operator+=(const Vector4D& rkV);
		Vector4D& operator-=(const Vector4D& rkV);
		Vector4D& operator*=(Real fScalar);
		Vector4D& operator/=(Real fScalar);

		Real& operator[] (int i);
		Real operator[] (int i) const;

		template<typename J>
		friend std::ostream& operator<<(std::ostream& out, const Vector4D<J>& vect);

		Real Length() const;
		Real SquaredLength() const;
		Real Dot(const Vector4D& rhs) const;

		void Normalize();

		bool Equals(const Vector4D& rhs) const;

		static const Vector4D ZERO;
		static const Vector4D UNITX;
		static const Vector4D UNITY;
		static const Vector4D UNITZ;
		static const Vector4D UNITW;
	};

	template<typename J>
	std::ostream& operator<<(std::ostream& out, const Vector4D<J>& vect)
	{
		out << vect.x << " " << vect.y << " " << vect.z << " " << vect.w;
		return out;
	}

	typedef Vector4D<int> Vector4Di;
	typedef Vector4D<float> Vector4Df;
	typedef Vector4D<double> Vector4Dd;

#include "Vector4D.inl"
}

#endif
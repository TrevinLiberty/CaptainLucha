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
 *	@file	Vector2D.h
 *	@brief	
 *
/****************************************************************************/

#ifndef VECTOR_2D_H_CL
#define VECTOR_2D_H_CL

#include "Utils/Utils.h"

namespace CaptainLucha
{
	template<class Real>
	class Vector2D
	{
	public:
		Vector2D() : x((Real)0.0), y((Real)0.0) {};
		Vector2D(Real x, Real y) : x(x), y(y) {};
		Vector2D(const Vector2D<Real>& vect): x(vect.x), y(vect.y) {};

		Real x;
		Real y;

		Vector2D& operator= (const Vector2D& rhs);

		bool operator== (const Vector2D& rhs) const;
		bool operator!= (const Vector2D& rhs) const;

		Vector2D operator+ (const Vector2D& rhs) const;
		Vector2D operator- (const Vector2D& rhs) const;
		Vector2D operator* (Real scalar) const;
		Vector2D operator/ (Real scalar) const;

		Vector2D& operator+= (const Vector2D& rhs);
		Vector2D& operator-= (const Vector2D& rhs);
		Vector2D& operator*= (Real scalar);
		Vector2D& operator/= (Real scalar);

		template<typename T>
		operator Vector2D<T>()
		{
			return Vector2D<T>(static_cast<T>(x), static_cast<T>(y));
		}

		float Length () const;
		Real SquaredLength () const;
		Real Dot (const Vector2D& rhs) const;

		void FastNormalize();
		void Normalize();
		
		Vector2D Reflect(const Vector2D& norm);

		Vector2D Perp() const;

		Vector2D GetRotate(float angle) const;

		template<typename T>
		Vector2D<T> Project(const Vector2D<T>& b)
		{
			return b * (this->Dot(b) / b.Dot(b));
		}

		bool Equals(const Vector2D& rhs) const;

		template<typename T>
		Vector2D<T> Get() const {return Vector2D<T>(static_cast<T>(x), static_cast<T>(y));}

		template<typename J>
		friend std::ostream& operator<<(std::ostream& out, const Vector2D<J>& vect);

		static const Vector2D ZERO;
		static const Vector2D UNITX;
		static const Vector2D UNITY;
	};

	template<typename J>
	std::ostream& operator<<(std::ostream& out, const Vector2D<J>& vect)
	{
		out << vect.x << " " << vect.y;
		return out;
	}

	typedef Vector2D<int> Vector2Di;
	typedef Vector2D<float> Vector2Df;
	typedef Vector2D<double> Vector2Dd;

#include "Vector2D.inl"
}

#endif
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
 *	@file	Math.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MATH_H_CL
#define MATH_H_CL

#include "Vector3D.h"
#include "Utils/CommonIncludes.h"

#include <cmath>

namespace CaptainLucha
{
	template<class Real>
	class Vector2D;

	template<class Real>
	class Math
	{
	public:
		static const Real PI;
		static const Real MAX_REAL;
		static const Real EPSILON;
		static const Real DEG_TO_RAD;
		static const Real RAD_TO_DEG;
	};

	typedef Math<float> MathF;
	typedef Math<double> MathD;

	template<typename Ta, typename Tb>
	bool FloatEquality(Ta v1, Tb v2, Ta Epsilon = Math<Ta>::EPSILON)
	{
		return std::fabs(v1 - v2) < Epsilon;
	}

	template<typename T>
	bool GetLineIntersection(
		const Vector2D<T>& p1, const Vector2D<T>& p2, 
		const Vector2D<T>& p3, const Vector2D<T>& p4, 
		Vector2D<T>& outVector)	
	{
		T x12 = p1.x - p2.x;
		T x34 = p3.x - p4.x;
		T y12 = p1.y - p2.y;
		T y34 = p3.y - p4.y;

		T c = x12 * y34 - y12 * x34;

		if (fabs(c) < 1)
		{
			// No intersection
			return false;
		}
		else
		{
			// Intersection
			T a = p1.x * p2.y - p1.y * p2.x;
			T b = p3.x * p4.y - p3.y * p4.x;

			outVector.x = (a * x34 - b * x12) / c;
			outVector.y = (a * y34 - b * y12) / c;

			return true;
		}
	}

	template<typename T>
	bool DoLinesIntersect(
		const Vector2D<T>& p1, const Vector2D<T>& p2, 
		const Vector2D<T>& p3, const Vector2D<T>& p4)	
	{
		float ud = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
		if (abs(ud) > 0.01) 
		{
			ud = 1 / ud;
			float t1 = (p1.y - p3.y);
			float t2 = (p1.x - p3.x);
			float ua = ((p4.x - p3.x) * t1 - (p4.y - p3.y) * t2) * ud;
			float ub = ((p2.x - p1.x) * t1 - (p2.y - p1.y) * t2) * ud;

			if (ua < 0 || ua > 1 || ub < 0 || ub > 1) 
				return false;
		}
		else
			return false;

		return true;
	}

	template<typename T>
	bool IsPointOnSimpleLine(const Vector2D<T>& p, const Vector2D<T>& l1, const Vector2D<T>& l2)
	{
		Vector2D<T> minimum(min(l1.x, l2.x) - (T)1, min(l1.y, l2.y) - (T)1);
		Vector2D<T> maximum(max(l1.x, l2.x) + (T)1, max(l1.y, l2.y) + (T)1);

		if(p.x >= minimum.x_ && p.x <= maximum.x_ && p.y >= minimum.y_ && p.y <= maximum.y_)
			return true;
		return false; 
	}

	inline void MakeOrthonormalBasis(Vector3D<Real>& x, Vector3D<Real>& y, Vector3D<Real>& z)
	{
		z = x.CrossProduct(y);
		if(FloatEquality(z.SquaredLength(), 0.0f))
			return;

		z.Normalize();
		y = z.CrossProduct(x);
	}
}

#endif
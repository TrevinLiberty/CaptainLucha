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

#include "Vector4D.h"
#include "Math.h"

#include <cmath>
using std::abs;

namespace CaptainLucha
{
	template<>
	bool Vector4D<float>::Equals(const Vector4D<float>& rhs) const
	{
		return (FloatEquality(x, rhs.x)
			&& FloatEquality(y, rhs.y)
			&& FloatEquality(z, rhs.z)
			&& FloatEquality(w, rhs.w));
	}

	template<>
	bool Vector4D<double>::Equals(const Vector4D<double>& rhs) const
	{
		return (FloatEquality(x, rhs.x)
			&& FloatEquality(y, rhs.y)
			&& FloatEquality(z, rhs.z)
			&& FloatEquality(w, rhs.w));
	}

	template<>
	bool Vector4D<int>::Equals(const Vector4D<int>& rhs) const
	{
		return (x == rhs.x 
			 && y == rhs.y 
			 && z == rhs.z 
			 && w == rhs.w);
	}

	template<> const Vector4D<float> Vector4D<float>::ZERO (0.0f, 0.0f, 0.0f, 0.0f);
	template<> const Vector4D<float> Vector4D<float>::UNITX(1.0f, 0.0f, 0.0f, 0.0f);
	template<> const Vector4D<float> Vector4D<float>::UNITY(0.0f, 1.0f, 0.0f, 0.0f);
	template<> const Vector4D<float> Vector4D<float>::UNITZ(0.0f, 0.0f, 1.0f, 0.0f);
	template<> const Vector4D<float> Vector4D<float>::UNITW(0.0f, 0.0f, 1.0f, 0.0f);

	template<> const Vector4D<double> Vector4D<double>::ZERO (0.0, 0.0, 0.0, 0.0);
	template<> const Vector4D<double> Vector4D<double>::UNITX(1.0, 0.0, 0.0, 0.0);
	template<> const Vector4D<double> Vector4D<double>::UNITY(0.0, 1.0, 0.0, 0.0);
	template<> const Vector4D<double> Vector4D<double>::UNITZ(0.0, 0.0, 1.0, 0.0);
	template<> const Vector4D<double> Vector4D<double>::UNITW(0.0, 0.0, 1.0, 0.0);

	template<> const Vector4D<int> Vector4D<int>::ZERO (0, 0, 0, 0);
	template<> const Vector4D<int> Vector4D<int>::UNITX(1, 0, 0, 0);
	template<> const Vector4D<int> Vector4D<int>::UNITY(0, 1, 0, 0);
	template<> const Vector4D<int> Vector4D<int>::UNITZ(0, 0, 1, 0);
	template<> const Vector4D<int> Vector4D<int>::UNITW(0, 0, 1, 0);
}
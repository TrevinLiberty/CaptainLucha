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

#include "Math.h"

#include "Vector2D.h"

#include <math.h>
#include <float.h>

namespace CaptainLucha
{
	template<> const float Math<float>::EPSILON = FLT_EPSILON;
	template<> const float Math<float>::MAX_REAL = FLT_MAX;
	template<> const float Math<float>::PI = (float)(4.0*atan(1.0));
	template<> const float Math<float>::DEG_TO_RAD = Math<float>::PI/180.0f;
	template<> const float Math<float>::RAD_TO_DEG = 180.0f/Math<float>::PI;

	template<> const double Math<double>::EPSILON = DBL_EPSILON;
	template<> const double Math<double>::MAX_REAL = DBL_MAX;
	template<> const double Math<double>::PI = 4.0*atan(1.0);
	template<> const double Math<double>::DEG_TO_RAD = Math<double>::PI/180.0;
	template<> const double Math<double>::RAD_TO_DEG = 180.0/Math<double>::PI;
}
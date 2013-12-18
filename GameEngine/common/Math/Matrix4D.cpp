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

namespace CaptainLucha
{
	template<> const Matrix4D<float> Matrix4D<float>::ZERO(
		0.0f, 0.0f, 0.0f, 0.0f, 
		0.0f, 0.0f, 0.0f, 0.0f, 
		0.0f, 0.0f, 0.0f, 0.0f, 
		0.0f, 0.0f, 0.0f, 0.0f);
	template<> const Matrix4D<float> Matrix4D<float>::IDENTITY(
		1.0f, 0.0f, 0.0f, 0.0f, 
		0.0f, 1.0f, 0.0f, 0.0f, 
		0.0f, 0.0f, 1.0f, 0.0f, 
		0.0f, 0.0f, 0.0f, 1.0f);

	template<> const Matrix4D<double> Matrix4D<double>::ZERO(
		0.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 0.0);
	template<> const Matrix4D<double> Matrix4D<double>::IDENTITY(
		1.0, 0.0, 0.0, 0.0, 
		0.0, 1.0, 0.0, 0.0, 
		0.0, 0.0, 1.0, 0.0, 
		0.0, 0.0, 0.0, 1.0);

	template<> const Matrix4D<int> Matrix4D<int>::ZERO(
		0, 0, 0, 0, 
		0, 0, 0, 0, 
		0, 0, 0, 0, 
		0, 0, 0, 0);
	template<> const Matrix4D<int> Matrix4D<int>::IDENTITY(
		1, 0, 0, 0, 
		0, 1, 0, 0, 
		0, 0, 1, 0, 
		0, 0, 0, 1);
}
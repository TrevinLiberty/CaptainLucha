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

#include "Color.h"

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	Vector4Di Color::GetColorAsVecInts() const
	{
		return Vector4Di((int)(r*255), (int)(g*255), (int)(b*255), (int)(a*255));
	}

	Vector4Df Color::GetColorAsVecFloats() const
	{
		return Vector4Df(r, g, b, a);
	}

	Color Color::Interpolate(const Color& color, float p) const
	{
		Color newColor;
		float np = 1.0f - p;

		newColor.r = r * p + color.r * np;
		newColor.g = g * p + color.g * np;
		newColor.b = b * p + color.b * np;
		newColor.a = a * p + color.a * np;
		return newColor;
	}

	float& Color::operator[] (int i)
	{
		return (&r)[i];
	}

	float Color::operator[] (int i) const
	{
		return (&r)[i];
	}

	//////////////////////////////////////////////////////////////////////////
	//	Static
	//////////////////////////////////////////////////////////////////////////
	const Color Color::Black(0.0f, 0.0f, 0.0f, 1.0f);
	const Color Color::White(1.0f, 1.0f, 1.0f, 1.0f);
	const Color Color::Grey(0.5f, 0.5f, 0.5f, 1.0f);
	const Color Color::Red(1.0f, 0.0f, 0.0f, 1.0f);
	const Color Color::Green(0.0f, 1.0f, 0.0f, 1.0f);
	const Color Color::Blue(0.0f, 0.0f, 1.0f, 1.0f);
	const Color Color::Yellow(1.0f, 1.0f, 0.0f, 1.0f);
	const Color Color::Pink(1.0f, 0.0f, 1.0f, 1.0f);
	const Color Color::Cyan(0.0f, 1.0f, 1.0f, 1.0f);
	const Color Color::Purple(0.5f, 0.0f, 1.0f, 1.0f);
	const Color Color::Lime(0.5f, 1.0f, 0.0f, 1.0f);
	const Color Color::Orange(1.0f, 0.5f, 0.0f, 1.0f);
	const Color Color::BrightPink(1.0f, 0.0f, 0.5f, 1.0f);
	const Color Color::LightGreen(0.0f, 1.0f, 0.5f, 1.0f);
	const Color Color::Transparent(0.0f, 0.0f, 0.0f, 0.0f);
}
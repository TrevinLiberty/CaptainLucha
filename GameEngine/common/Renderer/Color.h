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
 *	@file	Color.h
 *	@brief	
 *
/****************************************************************************/

#ifndef COLOR_H_CL
#define COLOR_H_CL

#include "Utils/UtilDebug.h"
#include "Math/Vector4D.h"

namespace CaptainLucha
{
	class Color
	{
	public:
		Color() {*this = Black;}
		Color(int r, int g, int b, int a = 255)
			: r(r/255.0f), g(g/255.0f), b(b/255.0f), a(a/255.0f) {}
		Color(const Vector4Di& vect)
			: r(vect.x/255.0f), g(vect.y/255.0f), b(vect.z/255.0f), a(vect.w/255.0f) {}
		Color(float r, float g, float b, float a = 1.0f)
			: r(r), g(g), b(b), a(a) {}
		Color(const Vector4Df& vect)
			: r(vect.x), g(vect.y), b(vect.z), a(vect.w) {}
		~Color() {};

		Vector4Di GetColorAsVecInts() const;
		Vector4Df GetColorAsVecFloats() const;

		Color Interpolate(const Color& color, float p) const;

		float& operator[] (int i);
		float operator[] (int i) const;

		float r;
		float g;
		float b;
		float a;

		static const Color Black;
		static const Color White;
		static const Color Grey;
		static const Color Red;
		static const Color Green;
		static const Color Blue;
		static const Color Yellow;//1 1 0
		static const Color Pink; // 1 0 1
		static const Color Cyan; //0 1 1
		static const Color Purple;//.5 0 1
		static const Color Lime;//.5 1 0
		static const Color Orange; //1 .5 0
		static const Color BrightPink; //1 0 .5
		static const Color LightGreen; //0 1 .5
		static const Color Transparent;
	};
}

#endif
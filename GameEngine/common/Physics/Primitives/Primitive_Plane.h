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
 *	@file	Primitive_Plane.h
 *	@brief	
 *
/****************************************************************************/

#ifndef PRIMITIVE_PLANE_H_CL
#define PRIMITIVE_PLANE_H_CL

#include "Primitive.h"

namespace CaptainLucha
{
	class Primitive_Plane : public Primitive
	{
	public:
		Primitive_Plane(const Vector3D<Real>& normal, Real offset)
			: Primitive(CL_Primitive_Plane),
			  m_normal(normal),
			  m_offset(offset)
		{}
		~Primitive_Plane();

		const Vector3D<Real>& GetNormal() const {return m_normal;}

		Real GetOffset() const {return m_offset;}

	protected:

	private:
		Vector3D<Real> m_normal;
		Real m_offset;
	};
}

#endif
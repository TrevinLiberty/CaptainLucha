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
 *	@file	Primitive.h
 *	@brief	
 *
/****************************************************************************/

#ifndef PRIMITIVE_H_CL
#define PRIMITIVE_H_CL

#include <Utils/UtilDebug.h>
#include <Math/Transformation.h>
#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	enum PrimitiveType
	{
		CL_Primitive_Box,
		CL_Primitive_Sphere,
		CL_Primitive_Plane
	};

	class RigidBody;

	class Primitive
	{
	public:
		Primitive(PrimitiveType type)
			: m_type(type)
		{}

		~Primitive() {};

		PrimitiveType GetType() const {return m_type;}

		Transformation& GetTransformation() {return m_transformation;}
		const Transformation& GetTransformation() const {return m_transformation;}
		void SetTransformation(const Transformation& val) {m_transformation = val;}

	protected:

	private:
		PrimitiveType m_type;

		Transformation m_transformation;
	};
}

#endif
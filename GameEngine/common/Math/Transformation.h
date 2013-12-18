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
 *	@file	Transformation.h
 *	@brief	
 *
/****************************************************************************/

#ifndef TRANSFORMATION_H_CL
#define TRANSFORMATION_H_CL

#include "Vector3D.h"
#include "Quaternion.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class Transformation
	{
	public:
		Transformation();
		~Transformation();

		const Vector3D<Real>& GetPosition() const {return m_position;}
		void SetPosition(const Vector3D<Real>& val) {m_position = val;}

		const CaptainLucha::Quaternion& GetOrientation() const {return m_orientation;}
		void SetOrientation(const CaptainLucha::Quaternion& val) {m_orientation = val;}

		Vector3D<Real> TransformPosition(const Vector3D<Real>& pos) const;
		Vector3D<Real> TransformDirection(const Vector3D<Real>& dir) const;

		Vector3D<Real> TransformTransposePosition(const Vector3D<Real>& pos) const;
		Vector3D<Real> TransformTransposeDirection(const Vector3D<Real>& dir) const;

	protected:

	private:
		Vector3D<Real> m_position;
		Quaternion m_orientation;
	};
}

#endif
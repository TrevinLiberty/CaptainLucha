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

#include "Transformation.h"

namespace CaptainLucha
{
	Transformation::Transformation() {}
	Transformation::~Transformation() {}

	Vector3D<Real> Transformation::TransformPosition(const Vector3D<Real>& pos) const
	{
		Matrix3D<Real> boxRot = m_orientation.GetMatrix();
		Vector3D<Real> result = boxRot * pos;
		return result + m_position;
	}

	Vector3D<Real> Transformation::TransformDirection(const Vector3D<Real>& dir) const
	{
		Matrix3D<Real> boxRot = m_orientation.GetMatrix();
		return boxRot * dir;
	}

	Vector3D<Real> Transformation::TransformTransposePosition(const Vector3D<Real>& pos) const
	{
		Matrix3D<Real> boxRot = m_orientation.GetMatrix().GetTranspose();
		Vector3D<Real> result = boxRot * (pos - m_position);
		return result;// + m_position;
	}

	Vector3D<Real> Transformation::TransformTransposeDirection(const Vector3D<Real>& dir) const
	{
		Matrix3D<Real> boxRot = m_orientation.GetMatrix().GetTranspose();
		return boxRot * dir;
	}
}
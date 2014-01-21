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
 *	@file	Plane.h
 *	@brief	
 *
/****************************************************************************/

#ifndef PLANE_H_CL
#define PLANE_H_CL

#include "Vector3D.h"

namespace CaptainLucha
{
	struct Plane
	{
        Plane() {};
        Plane(
            const Vector3Df& point, 
            const Vector3Df& normal) 
        {SetNormalAndPoint(normal, point);}
        ~Plane() {};

        inline void SetNormalAndPoint(
            const Vector3Df& normal, 
            const Vector3Df& point)
        {
            m_normal = normal;
            m_d = (-normal).Dot(point);
        }

        inline float Distance(const Vector3Df& point) const
        {
            return m_normal.Dot(point) + m_d;
        }

        float m_d;
        Vector3Df m_normal;
	};
}

#endif
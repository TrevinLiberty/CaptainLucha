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
 *	@file	ViewFrustum.h
 *	@brief	
 *
/****************************************************************************/

#ifndef VIEWFRUSTUM_H_CL
#define VIEWFRUSTUM_H_CL

#include "Math/Plane.h"

namespace CaptainLucha
{
	class ViewFrustum
	{
	public:
        ViewFrustum() {};
        ~ViewFrustum() {};

        enum PlaneDirec{
            TOP = 0, BOTTOM, LEFT,
            RIGHT, NEARP, FARP
        };

        const Plane& GetPlane(int index) const {return m_frustumPlanes[index];}

        void UpdateData(float fovDegree, float aspectRatio, float zNear, float zFar);

        void UpdateFrustum(const Vector3Df& cameraPos, const Vector3Df& cameraDir);

        bool PointInFrustum(const Vector3Df& point) const;
        bool SphereInFrustum(const Vector3Df& point, float radius) const;
        bool AABBInFrustum(const Vector3Df& min, const Vector3Df& max) const;

        //outPoints must be an array of atlease size 8
        void GetFrustumPoints(
            const Vector3Df& cameraPos, 
            const Vector3Df& cameraDir,
            Vector3Df* outPoints);

	private:
        Plane m_frustumPlanes[6];

        float m_zNear;
        float m_zFar;
        float m_fov;
        float m_aspectRatio;
	};
}

#endif
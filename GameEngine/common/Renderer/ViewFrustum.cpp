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
 *	@file	ViewFrustum.cpp
 *	@brief	
 *
/****************************************************************************/

#include "ViewFrustum.h"
#include "Utils/Utils.h"

namespace CaptainLucha
{

    void ViewFrustum::UpdateData(float fovDegree, float aspectRatio, float zNear, float zFar)
    {
        m_zNear = zNear;
        m_zFar = zFar;
        m_fov = fovDegree;
        m_aspectRatio = aspectRatio;
    }

    void ViewFrustum::UpdateFrustum(const Vector3Df& cameraPos, const Vector3Df& cameraDir)
    {
        const float FOVRadians = DegreesToRadians(m_fov * 0.5f);
        float tang = tan(FOVRadians);

        Vector3Df nearDim;
        nearDim.y = tang * m_zNear;
        nearDim.x = nearDim.y * m_aspectRatio;

        Vector3Df farDim;
        farDim.y = tang * m_zFar;
        farDim.x = farDim.y * m_aspectRatio;

        Vector3Df axisZ = cameraDir * -1;
        Vector3Df axisX = axisZ.CrossProduct(Vector3Df(0.0f, 1.0f, 0.0f)); axisX.Normalize();
        Vector3Df axisY = axisZ.CrossProduct(axisX); axisY.Normalize();

        Vector3Df nearP = cameraPos - axisZ * m_zNear;
        Vector3Df farP  = cameraPos - axisZ * m_zFar;

        m_frustumPlanes[NEARP].SetNormalAndPoint(-axisZ, nearP);
        m_frustumPlanes[FARP].SetNormalAndPoint(axisZ,   farP);

        Vector3Df temp, normal;

        temp = (nearP + axisY*nearDim.y) - cameraPos; temp.Normalize();
        normal = temp.CrossProduct(axisX);
        m_frustumPlanes[TOP].SetNormalAndPoint(normal, nearP + axisY * nearDim.y);

        temp = (nearP - axisY*nearDim.y) - cameraPos; temp.Normalize();
        normal = axisX.CrossProduct(temp);
        m_frustumPlanes[BOTTOM].SetNormalAndPoint(normal, nearP - axisY * nearDim.y);

        temp = (nearP - axisX*nearDim.x) - cameraPos; temp.Normalize();
        normal = temp.CrossProduct(axisY);
        m_frustumPlanes[LEFT].SetNormalAndPoint(normal, nearP - axisX * nearDim.x);

        temp = (nearP + axisX*nearDim.x) - cameraPos; temp.Normalize();
        normal = axisY.CrossProduct(temp);
        m_frustumPlanes[RIGHT].SetNormalAndPoint(normal, nearP + axisX * nearDim.x);
    }

    bool ViewFrustum::PointInFrustum(const Vector3Df& point) const
    {
        if(m_frustumPlanes[0].Distance(point) < 0) return false;
        if(m_frustumPlanes[1].Distance(point) < 0) return false;
        if(m_frustumPlanes[2].Distance(point) < 0) return false;
        if(m_frustumPlanes[3].Distance(point) < 0) return false;
        if(m_frustumPlanes[4].Distance(point) < 0) return false;
        if(m_frustumPlanes[5].Distance(point) < 0) return false;

        return true;
    }

    bool ViewFrustum::SphereInFrustum(const Vector3Df& point, float radius) const
    {
        if(m_frustumPlanes[0].Distance(point) < -radius) return false;
        if(m_frustumPlanes[1].Distance(point) < -radius) return false;
        if(m_frustumPlanes[2].Distance(point) < -radius) return false;
        if(m_frustumPlanes[3].Distance(point) < -radius) return false;
        if(m_frustumPlanes[4].Distance(point) < -radius) return false;
        if(m_frustumPlanes[5].Distance(point) < -radius) return false;

        return true;
    }

    bool ViewFrustum::AABBInFrustum(const Vector3Df& min, const Vector3Df& max) const
    {
        for(int i = 0; i < 6; ++i)
        {
            int numIN  = 0;

            if(m_frustumPlanes[i].Distance(Vector3Df(min.x, min.y, min.z)) >= 0) ++numIN;
            if(m_frustumPlanes[i].Distance(Vector3Df(min.x, min.y, max.z)) >= 0) ++numIN; 
            if(m_frustumPlanes[i].Distance(Vector3Df(min.x, max.y, min.z)) >= 0) ++numIN; 
            if(m_frustumPlanes[i].Distance(Vector3Df(min.x, max.y, max.z)) >= 0) ++numIN; 
            if(m_frustumPlanes[i].Distance(Vector3Df(max.x, min.y, min.z)) >= 0) ++numIN; 
            if(m_frustumPlanes[i].Distance(Vector3Df(max.x, min.y, max.z)) >= 0) ++numIN; 
            if(m_frustumPlanes[i].Distance(Vector3Df(max.x, max.y, min.z)) >= 0) ++numIN; 
            if(m_frustumPlanes[i].Distance(Vector3Df(max.x, max.y, max.z)) >= 0) ++numIN; 

            if(numIN == 0)
                return false;
        }

        return true;
    }

    void ViewFrustum::GetFrustumPoints(
        const Vector3Df& cameraPos, 
        const Vector3Df& cameraDir,
        Vector3Df* outPoints)
    {
        const float FOVRadians = DegreesToRadians(m_fov * 0.5f);
        float tang = tan(FOVRadians);

        Vector3Df nearD;
        nearD.y = tang * m_zNear;
        nearD.x = nearD.y * m_aspectRatio;

        Vector3Df farD;
        farD.y = tang * m_zFar;
        farD.x = farD.y * m_aspectRatio;

        Vector3Df Z = cameraDir * -1;
        Vector3Df X = Z.CrossProduct(Vector3Df(0.0f, 1.0f, 0.0f));
        X.Normalize();
        Vector3Df Y = Z.CrossProduct(X);
        Y.Normalize();

        Vector3Df nc = cameraPos - Z * m_zNear;
        Vector3Df fc = cameraPos - Z * m_zFar;

        float nh = nearD.y;
        float nw = nearD.x;
        float fh = farD.y;
        float fw = farD.x;

        outPoints[0] = nc + Y * nh - X * nw;
        outPoints[1] = nc + Y * nh + X * nw;
        outPoints[2] = nc - Y * nh - X * nw;
        outPoints[3] = nc - Y * nh + X * nw;
                      
        outPoints[4] = fc + Y * fh - X * fw;
        outPoints[5] = fc + Y * fh + X * fw;
        outPoints[6] = fc - Y * fh - X * fw;
        outPoints[7] = fc - Y * fh + X * fw;
    }
}
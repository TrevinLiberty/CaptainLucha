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

#include "MatrixStack.h"

#include "Utils/UtilDebug.h"
#include "Utils/Utils.h"

#define _USE_MATH_DEFINES
#include <Math.h>

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	MatrixStack::MatrixStack()
		: m_isPerspectiveInit(false),
		  m_isOrthoInit(false),
		  m_currentProjectionMode(CL_PROJECTION)
	{
		LoadIdentity();
	}

	MatrixStack::~MatrixStack()
	{

	}

	void MatrixStack::LoadIdentity()
	{
		PopMatrix();
		m_stack.push(Matrix4Df::IDENTITY);
	}

	void MatrixStack::LoadMatrix(const Matrix4Df& matrix)
	{
		PopMatrix();
		m_stack.push(matrix);
	}

	void MatrixStack::PushMatrix()
	{
		m_stack.push(m_stack.top());
	}

	void MatrixStack::PopMatrix()
	{
		if(!m_stack.empty())
			m_stack.pop();
	}

	void MatrixStack::MultMatrix(const Matrix4Df& matrixRHS)
	{
		Matrix4Df temp(matrixRHS * m_stack.top());
		PopMatrix();
		m_stack.push(temp);
	}

	void MatrixStack::Rotate(float degrees, float x, float y, float z)
	{
		RotateRad(degrees * (float)M_PI / 180.0f, x, y, z);
	}

	void MatrixStack::Rotate(float degrees, const Vector3Df& axis)
	{
		Rotate(degrees, axis.x, axis.y, axis.z);
	}

	void MatrixStack::RotateRad(float radians, float x, float y, float z)
	{
		MultMatrix(Matrix4Df(radians, x, y, z));
	}

	void MatrixStack::RotateRad(float radians, const Vector3Df& axis)
	{
		Rotate(radians, axis.x, axis.y, axis.z);
	}

	void MatrixStack::Translate(float x, float y, float z)
	{
		MultMatrix(Matrix4Df(x, y, z));
	}

	void MatrixStack::Translate(const Vector3Df& trans)
	{
		Translate(trans.x, trans.y, trans.z);
	}

	void MatrixStack::Scale(float x, float y, float z)
	{
		MultMatrix(Matrix4Df
			(x,	   0.0f, 0.0f, 0.0f,
			 0.0f, y,	 0.0f, 0.0f,
			 0.0f, 0.0f, z,	   0.0f,
			 0.0f, 0.0f, 0.0f, 1.0f));
	}

	const Matrix4Df& MatrixStack::GetModelMatrix() const
	{
		return m_stack.top();
	}

	const Matrix4Df& MatrixStack::GetViewMatrix() const 
	{
		return m_currentProjectionMode == CL_PROJECTION ? m_view : Matrix4Df::IDENTITY;
	}

	const Matrix4Df& MatrixStack::GetProjectionMatrix() const 
	{
		return m_currentProjectionMode == CL_PROJECTION ? m_projectionMatrix : m_orthoMatrix;
	}

	Matrix4Df MatrixStack::GetModelViewMatrix() const
	{
		return m_view * m_stack.top();
	}

	Matrix4Df MatrixStack::GetModelViewProjectionMatrix() const
	{
		return (m_currentProjectionMode == CL_PROJECTION ? m_stack.top() * m_view * m_projectionMatrix : m_stack.top() * m_orthoMatrix);
	}

	void MatrixStack::Perspective(float fov, float aspect, float zNear, float zFar)
	{
        const float FOVRadians = DegreesToRadians(fov * 0.5f);
		float maxY = zNear * tanf(FOVRadians);
		float maxX = maxY * aspect;
		Frustum(-maxX, maxX, -maxY, maxY, zNear, zFar);

        m_frustum.UpdateData(fov, aspect, zNear, zFar);
        m_frustum.UpdateFrustum(Vector3Df(), Vector3Df(0.0f, 0.0f, -1.0f));
	}

	void MatrixStack::Othographic(int left, int right, int bottom, int top, int zFar, int zNear)
	{
		float t2 = static_cast<float>(right - left);
		float t3 = static_cast<float>(top - bottom);
		float t4 = (float)(zFar - zNear);

		m_orthoMatrix = Matrix4Df::ZERO;

		m_orthoMatrix[0] = 2.0f / t2;
		m_orthoMatrix[5] = 2.0f / t3;
		m_orthoMatrix[10] = -2.0f / t4;
		m_orthoMatrix[12] = -((right + left) / t2);
		m_orthoMatrix[13] = -((top + bottom) / t3);
		m_orthoMatrix[14] = (float)(-((zFar + zNear) / (zFar - zNear)));
		m_orthoMatrix[15] = 1.0f;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void MatrixStack::Frustum(float left, float right, float bottom, float top, float zNear, float zFar)
	{
		float t1 = 2.0f * zNear;
		float t2 = right - left;
		float t3 = top - bottom;
		float t4 = zFar - zNear;

		m_projectionMatrix = Matrix4Df::ZERO;

		m_projectionMatrix[0] = t1 / t2;
		m_projectionMatrix[5] = t1 / t3;
		m_projectionMatrix[8] = (right + left) / t2;
		m_projectionMatrix[9] = (top + bottom) / t3;
		m_projectionMatrix[10] = (-zFar - zNear) / t4;
		m_projectionMatrix[11] = -1;
		m_projectionMatrix[14] = (-t1 * zFar) / t4;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Private
	//////////////////////////////////////////////////////////////////////////
	MatrixStack* MatrixStack::m_matrixStack = NULL;
}
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
 *	@file	MatrixStack.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MATRIX_STACK_H_CL
#define MATRIX_STACK_H_CL

#include "Math/Matrix4D.h"
#include "Math/Vector2D.h"
#include "ViewFrustum.h"
#include "Utils/UtilMacros.h"

#include <stack>

namespace CaptainLucha
{
	enum ProjectionMode
	{
		CL_PROJECTION,
		CL_ORTHOGRAPHIC
	};

	class MatrixStack
	{
	public:
		MatrixStack();
		~MatrixStack();

		void LoadIdentity();
		void LoadMatrix(const Matrix4Df& matrix);

		void PushMatrix();
		void PopMatrix();

		void MultMatrix(const Matrix4Df& matrixRHS);

		void Rotate(float degrees, float x, float y, float z);
		void Rotate(float degrees, const Vector3Df& axis);

		void RotateRad(float radians, float x, float y, float z);
		void RotateRad(float radians, const Vector3Df& axis);

		void Translate(float x, float y, float z);
		void Translate(const Vector3Df& trans);

		void Scale(float x, float y, float z);

		void SetViewMatrix(const Matrix4Df& view) {m_view = view;}

		const Matrix4Df& GetModelMatrix() const;
		const Matrix4Df& GetViewMatrix() const;
		const Matrix4Df& GetProjectionMatrix() const;
	
		Matrix4Df GetModelViewMatrix() const;
		Matrix4Df GetModelViewProjectionMatrix() const;

		void Perspective(float fov, float aspect, float zNear, float zFar);
		void Othographic(int left, int right, int bottom, int top, int zFar, int zNear);

		//Sets the projection matrix used for drawing
		void SetProjectionMode(ProjectionMode mode) {m_currentProjectionMode = mode;}
		ProjectionMode GetCurrentProjectionMode() const {return m_currentProjectionMode;}

        void UpdateFrustum(const Vector3Df& viewPos, const Vector3Df& viewDir) 
        {
            m_frustum.UpdateFrustum(viewPos, viewDir);
        }

        const ViewFrustum& GetViewFrustum() const {return m_frustum;}    
        ViewFrustum& GetViewFrustum() {return m_frustum;} 

	protected:
		void Frustum(float left, float right, float bottom, float top, float zNear, float zFar);

	private:
		static MatrixStack* m_matrixStack;

		Matrix4Df m_orthoMatrix;
		Matrix4Df m_projectionMatrix;

		Matrix4Df m_view;

        ViewFrustum m_frustum;

		std::stack<Matrix4Df> m_stack;

		bool m_isPerspectiveInit;
		bool m_isOrthoInit;
		
		ProjectionMode m_currentProjectionMode;

		PREVENT_COPYING(MatrixStack);
	};
}

#endif
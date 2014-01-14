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
 *	@file	Renderer.h
 *	@brief	
 *
/****************************************************************************/

#ifndef RENDERER_H_CL
#define RENDERER_H_CL

#include "Math/Matrix4D.h"
#include "Math/Vector3D.h"

#include "Lights/Light.h"
#include "Lights/Light_Spot.h"

#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class Object;
	class GLProgram;

	class Renderer
	{
	public:
		Renderer();
		virtual ~Renderer();

		virtual void Draw() = 0;

		/**
		 * @brief     Adds object to the renderer to enable drawing.
		 * @param	  Object * object
		 */
		void AddObjectToRender(Object* object);

		/**
		 * @brief     Removes the object from renderer.
		 * @param	  Object * object
		 */
		void RemoveObject(Object* object);

		/**
		 * @brief     Sets the view matrix for the next call to DeferredRenderer:Draw();
		 * @param	  const Matrix4Df & view
		 */
		void SetViewMatrix(const Matrix4Df& view) {m_viewMatrix = view;}

		/**
		 * @brief     Sets the current position of the camera. Used for specular shading
		 * @param	  const Vector3Df & camPos
		 */
		void SetCameraPos(const Vector3Df& camPos) {m_cameraPos = camPos;}

		virtual Light*				CreateNewPointLight() = 0;
		virtual Light*				CreateNewAmbientLight() = 0;
		virtual Light_Directional*	CreateNewDirectionalLight() = 0;
		virtual Light_Spot*			CreateNewSpotLight() = 0;

		/**
		 * @brief		Removes a light created for the renderer.
		 * @param		DeferredLight * light
		 * @attention	This deletes the light if found.
		 */
		virtual void RemoveLight(Light* light) = 0;

		void SetDebugDrawing(bool t) {m_debugDraw = t;}

	protected:
		virtual void DrawScene(GLProgram& program);

		bool m_debugDraw;

		Matrix4Df m_viewMatrix;
		Vector3Df m_cameraPos;

		std::vector<Object*> m_renderableObjects;
	};
}

#endif
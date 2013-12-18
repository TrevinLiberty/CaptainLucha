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
 *	@file	DeferredRenderer.h
 *	@brief	Deferred renderer responsible for drawing all objects added by DeferredRenderer::AddObjectToRender(Object* object),
 *				and applying all DeferredLight.
 *  @see	DeferredLight
 *
/****************************************************************************/

#ifndef DEFERREDRENDERER_H_CL
#define DEFERREDRENDERER_H_CL

#include <Math/Vector3D.h>
#include <Math/Matrix4D.h>
#include <Renderer/RendererUtils.h>
#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class Object;
	class GLProgram;
	class GLTexture;
	class DeferredLight;
	class DeferredLight_Point;
	class DeferredLight_Ambient;
	class DeferredLight_Directional;
	class DeferredLight_Spot;
	class FrameBufferObject;
	class GBuffer;

	/**
	* @brief     Draw the whole scene.
	*/
	class DeferredRenderer
	{
	public:
		DeferredRenderer();
		virtual ~DeferredRenderer();

		/**
		 * @brief     Draw the whole scene.
		 */
		virtual void Draw();

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

		DeferredLight_Point* CreateNewPointLight();
		DeferredLight_Ambient* CreateNewAmbientLight();
		DeferredLight_Directional* CreateNewDirectionalLight();
		DeferredLight_Spot* CreateNewSpotLight();


		/**
		 * @brief		Removes a light created for the renderer.
		 * @param		DeferredLight * light
		 * @attention	This deletes the light if found.
		 */
		void RemoveLight(DeferredLight* light);

		/**
		 * @brief     Samples the last drawn scene to find the objectID of the center of the screen. The object ID is unique for every Object.
		 * @see		  Object.h
		 */
		int GetObjectIDFromRT() const;

		void SetDebugDrawing(bool t) {m_debugDraw = t;}

	protected:
		/**
		 * @brief     Adds a light created by CreateNewPointLight and CreateNewSpotLight.
		 * @param	  DeferredLight * newLight
		 */
		void AddNewLight(DeferredLight* newLight);

		/**
		 * @brief     Adds a light created by CreateNewAmbientLight and CreateNewDirectionalLight.
		 * @param	  DeferredLight * newLight
		 */
		void AddNewFullscreenLight(DeferredLight* newLight);

		/**
		 * @brief     Iterates through all added objects and draws them.
		 * @param	  GLProgram & program
		 */
		void DrawScene(GLProgram& program);

		/**
		 * @brief     DrawScene onto the render targets 0 - 4. 
		 */
		void PopulateGBuffers();


		/**
		 * @brief     Draws the final shaded scene to the screen. This include material color, diffuse light, and specular light.
		 */
		void RenderAccumulator();

		/**
		 * @brief     Iterates through all lights, accumulating their effect on the scene.
		 * @attention Should call BeginLightPass before and EndLightPass after.
		 */
		void LightPass();

		/**
		 * @brief     Sets OpenGL up for LightPass.
		 */
		void BeginLightPass();

		/**
		 * @brief     Resets OpenGL after LightPass.
		 */
		void EndLightPass();

		/**
		 * @brief     FBO Validation check
		 */
		bool ValidateFBO();


		/**
		 * @brief     Clears the FBO
		 */
		void ClearFBO();

		/**
		 * @brief     Generates all render targets and FBOs needed.
		 * @return    void
		 */
		void InitRenderer();

		/**
		 * @brief     Renders a full screen quad with texture coordinates.
		 */
		void FullScreenPass();

		void DebugRenderGBuffer();

		///////////////////////////////////////////////////////
		// Variables
		std::vector<Object*> m_renderableObjects;

		std::vector<DeferredLight*> m_lights;
		std::vector<DeferredLight*> m_fullscreenLights;

		bool m_debugDraw;

		Matrix4Df m_viewMatrix;
		Vector3Df m_cameraPos;

		//Deferred Shading
		unsigned int m_fbo0;
		unsigned int m_fbo1;

		GLTexture* m_rt0;
		GLTexture* m_rt1;
		GLTexture* m_rt2;
		GLTexture* m_rt3;
		GLTexture* m_accumDiffuseLightTexture;
		GLTexture* m_accumSpecularLightTexture;
		GLTexture* m_depth;

		FrameBufferObject* m_depthBuffer;

		GLProgram* m_depthProgram;
		GLProgram* m_debugDepthProgram;
		GLProgram* m_graphicsBufferProgram;
		GLProgram* m_finalPassProgram;
	};
}

#endif
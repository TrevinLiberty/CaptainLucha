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

#include "Geometry/SkyBox.h"

#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class Renderable;
	class GLProgram;

    typedef void (*DrawFunction)(GLProgram& program, void* userData);
    typedef std::pair<DrawFunction, void*> DrawPair;

    enum CLCurrentPass
    {
        CL_DRAW_PASS,
        CL_REFLECT_PASS,
        CL_SHADOW_PASS,
        CL_ALPHA_PASS
    };

	class Renderer
	{
	public:
		Renderer();
		virtual ~Renderer();

		virtual void Draw() = 0;

        /**
         * @brief     Draws the skybox if it exists. 
                        Assumes camera position is set.
         */
        void DrawSkybox();

		/**
		 * @brief     Adds object to the renderer to enable drawing.
		 * @param	  Object * object
		 */
		void AddObjectToRender(Renderable* object);

		/**
		 * @brief     Removes the object from renderer.
		 * @param	  Object * object
		 */
		void RemoveObject(Renderable* object);

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
        void SetCameraDir(const Vector3Df& camDir) {m_cameraDir = camDir;}

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

        void SetDebugDrawFunction(DrawFunction drawFunc, void* userData = NULL) 
        {
            m_debugDrawFunction = drawFunc;
            m_debugDrawUserData = userData;
        }

        void AddDrawFunction(DrawFunction func, void* userData = NULL);
        void RemoveDrawFunction(DrawFunction func, void* userData = NULL);

        const SkyBox* GetSkyBox() const {return m_skyBox;}
        void SetSkyBox(SkyBox* val) {m_skyBox = val;}

        //Valid at the end of CL_DRAW_PASS. 
        //Used for reflective surface to sample from, 
        //  since the reflective surface is drawn last.
        //
        //TODO: can be used for post processing passes.  
        GLTexture* GetFinalSceneTexture() {return m_finalRenderSceneTex;}

        virtual void SetReflectiveSurface(Renderable* surfaceObject, int width, int height);
        void SetRefelctiveInfo(const Vector3Df& normal, const Vector3Df& pos);

        //Only a valid texture during CL_DRAW_PASS and when a reflective 
        //  surface is set.
        GLTexture* GetReflectiveTexture() {return m_reflectionTexture;}
        bool HasReflectiveObject() {return m_reflectiveObject != NULL;}

        Vector4Df GetReflectiveClipPlane() const;

        CLCurrentPass GetCurrentPass() const {return m_currentPass;}

	protected:
		virtual void DrawScene(GLProgram& program, bool isAlphaPass);

		Matrix4Df m_viewMatrix;
		Vector3Df m_cameraPos;
        Vector3Df m_cameraDir;

        DrawFunction m_debugDrawFunction;
        void*        m_debugDrawUserData;

        GLTexture* m_finalRenderSceneTex;
        SkyBox* m_skyBox;
    
        std::vector<DrawPair> m_drawFunctions;
        std::vector<Renderable*> m_renderableObjects;

        CLCurrentPass m_currentPass;

        bool m_debugDraw;

        //Reflective Variables
        //
        unsigned int m_reflectiveFBO;
        Renderable* m_reflectiveObject;
        GLTexture*  m_reflectionTexture;

        int m_reflectiveWidth;
        int m_reflectiveHeight;

        Vector3Df   m_reflectiveNormal;
        Vector3Df   m_reflectivePos;
	};
}

#endif
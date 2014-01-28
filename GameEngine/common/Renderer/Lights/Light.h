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
 *	@file	Light.h
 *	@brief	
 *
/****************************************************************************/

#ifndef LIGHT_H_CL
#define LIGHT_H_CL

#include "Renderer/Color.h"
#include "Math/Vector3D.h"
#include "Renderer/Shader/GLProgram.h"

#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	enum LightType
	{
		CL_AMBIENT_LIGHT,
		CL_POINT_LIGHT,
		CL_DIRECTIONAL_LIGHT,
		CL_SPOT_LIGHT
	};

    class GLTexture;

	class Light
	{
	public:
		Light(LightType type);
		virtual ~Light();

		void SetRadius(float radius) {m_radius = radius;}
		float GetRadius() const {return m_radius;}

		void SetIntensity(float intensity) {m_intensity = intensity;}
		float GetIntensity() const {return m_intensity;}

		void SetPosition(const Vector3Df& pos) {m_position = pos;}
		const Vector3Df& GetPosition() const {return m_position;}

		void SetColor(const Color& color) {m_color = color;}
		const Color& GetColor() const {return m_color;}

		LightType GetType() const {return m_type;}

        
        /**
         * @brief     Virtual function for deferred lights
         */
        virtual void ApplyLight(
            const Vector3Df& cameraPos, 
            GLTexture* renderTarget0, 
            GLTexture* renderTarget1, 
            GLTexture* renderTarget2) 
        {UNUSED(cameraPos) UNUSED(renderTarget0) 
        UNUSED(renderTarget1) UNUSED(renderTarget2)};

	protected:
		LightType m_type;

		float	  m_radius;
		float	  m_intensity;
		Vector3Df m_position;
		Color	  m_color;

        static GLProgram* m_nullProgram;
	};
}

#endif
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
 *	@file	DeferredLight_Spot.h
 *	@brief	
 *
/****************************************************************************/

#ifndef DEFERREDLIGHT_SPOT_H_CL
#define DEFERREDLIGHT_SPOT_H_CL

#include "Renderer/Lights/DeferredLight.h"

namespace CaptainLucha
{
	class GLProgram;
	class GLTexture;

	class DeferredLight_Spot : public DeferredLight
	{
	public:
		DeferredLight_Spot();
		~DeferredLight_Spot();

		void ApplyLight(const Vector3Df& cameraPos, GLTexture* renderTarget0, GLTexture* renderTarget1, GLTexture* renderTarget2, GLTexture* renderTarget3);
		void StencilPass();

		void SetRadius(float radius) {m_radius = radius;}
		float GetRadius() const {return m_radius;}

		void SetLookAt(const Vector3Df& lookAt);

		const float GetInnerConeAngle() const {return m_innerConeAngle;}
		void SetInnerConeAngle(const float& degrees) {m_innerConeAngle = DegreesToRadians(degrees);}

		const float GetOuterConeAngle() const {return m_outerConeAngle;}
		void SetOuterConeAngle(const float& degrees) {m_outerConeAngle = DegreesToRadians(degrees);}

		const Vector3Df& GetLightDir() const {return m_lightDir;}
		void SetLightDir(const Vector3Df& val) {m_lightDir = val;}

	protected:

	private:
		float m_radius;

		float m_innerConeAngle;
		float m_outerConeAngle;

		Vector3Df m_lightDir;

		static GLProgram* m_glProgram;
	};
}

#endif
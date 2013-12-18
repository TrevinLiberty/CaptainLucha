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
 *	@file	DeferredLight.h
 *	@brief	
 *
/****************************************************************************/

#ifndef DEFERREDLIGHT_H_CL
#define DEFERREDLIGHT_H_CL

#include "Objects/Object.h"
#include "Renderer/Color.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	enum LightType
	{
		CL_AMBIENT_LIGHT,
		CL_POINT_LIGHT
	};

	class DeferredLight : public Object
	{
	public:
		DeferredLight(LightType type);
		~DeferredLight() {};

		virtual void ApplyLight(const Vector3Df& cameraPos, GLTexture* renderTarget0, GLTexture* renderTarget1, GLTexture* renderTarget2, GLTexture* renderTarget3) = 0;
		virtual void StencilPass() {}

		void SetObjectColor(const Color& c) {m_color = c;}
		void SetIntensity(float intensity) {m_intensity = intensity;}

		LightType GetType() const {return m_type;}

	protected:
		Color m_color;
		float m_intensity;

		static GLProgram* m_nullProgram;

	private:

		LightType m_type;
	};
}

#endif
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
 *	@file	DeferredLight_Directional.h
 *	@brief	
 *
/****************************************************************************/

#ifndef DEFERREDLIGHT_DIRECTIONAL_H_CL
#define DEFERREDLIGHT_DIRECTIONAL_H_CL

#include "Renderer/Lights/DeferredLight.h"

namespace CaptainLucha
{
	class GLProgram;
	class GLTexture;

	class DeferredLight_Directional : public DeferredLight
	{
	public:
		DeferredLight_Directional();
		~DeferredLight_Directional();

		void ApplyLight(const Vector3Df& cameraPos, GLTexture* renderTarget0, GLTexture* renderTarget1, GLTexture* renderTarget2, GLTexture* renderTarget3);

		const Vector3Df& GetLightDir() const {return m_lightDir;}
		void SetLightDir(const Vector3Df& val) {m_lightDir = val;}

	protected:

	private:
		static GLProgram* m_glProgram;

		Vector3Df m_lightDir;
	};
}

#endif
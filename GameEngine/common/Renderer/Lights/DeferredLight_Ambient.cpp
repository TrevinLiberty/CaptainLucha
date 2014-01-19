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

#include "DeferredLight_Ambient.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Texture/GLTexture.h"

namespace CaptainLucha
{
	DeferredLight_Ambient::DeferredLight_Ambient()
		: Light(CL_AMBIENT_LIGHT)
	{
		if(!m_glProgram)
			m_glProgram = new GLProgram("Data/Shaders/DeferredVertShader.vert", "Data/Shaders/DeferredAmbientLightShader.frag");
	}

	DeferredLight_Ambient::~DeferredLight_Ambient()
	{

	}

	void DeferredLight_Ambient::ApplyLight(const Vector3Df& cameraPos, GLTexture* renderTarget0, GLTexture* renderTarget1, GLTexture* renderTarget2)
	{
		UNUSED(cameraPos)

		g_MVPMatrix->SetProjectionMode(CL_ORTHOGRAPHIC);

		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->LoadIdentity();

		SetGLProgram(m_glProgram);

		SetTexture("renderTarget0", renderTarget0);
		SetTexture("renderTarget1", renderTarget1);
		SetTexture("renderTarget2", renderTarget2);

		SetUniform("lightColor", m_color);
		SetUniform("intensity", m_intensity);

		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f,			 0.0f,		     0.0f);
			clVertex3(WINDOW_WIDTHf, 0.0f,		     0.0f);
			clVertex3(WINDOW_WIDTHf, WINDOW_HEIGHTf, 0.0f);
			clVertex3(0.0f,		     WINDOW_HEIGHTf, 0.0f);
		}
		DrawEnd();

		SetGLProgram(NULL);

		g_MVPMatrix->SetProjectionMode(CL_PROJECTION);
		g_MVPMatrix->PopMatrix();
	}

	GLProgram* DeferredLight_Ambient::m_glProgram = NULL;
}
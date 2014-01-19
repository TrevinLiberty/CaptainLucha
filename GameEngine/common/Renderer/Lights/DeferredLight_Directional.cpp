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

#include "DeferredLight_Directional.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Texture/GLTexture.h"

namespace CaptainLucha
{
	DeferredLight_Directional::DeferredLight_Directional()
	{
		if(!m_glProgram)
			m_glProgram = new GLProgram("Data/Shaders/DeferredVertShader.vert", "Data/Shaders/DeferredDirectionLight.frag");
	}

	DeferredLight_Directional::~DeferredLight_Directional()
	{

	}

	void DeferredLight_Directional::ApplyLight(const Vector3Df& cameraPos, GLTexture* renderTarget0, GLTexture* renderTarget1, GLTexture* renderTarget2)
	{
		UNUSED(cameraPos)

		g_MVPMatrix->SetProjectionMode(CL_ORTHOGRAPHIC);

		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->LoadIdentity();

		SetGLProgram(m_glProgram);

        SetUtilsColor(m_color);

		SetTexture("renderTarget0", renderTarget0);
		SetTexture("renderTarget1", renderTarget1);
		SetTexture("renderTarget2", renderTarget2);

		SetUniform("intensity", m_intensity);
		SetUniform("lightDir", m_lightDir);

		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f,					0.0f,				 0.0f);
			clVertex3((float)WINDOW_WIDTH,	0.0f,				 0.0f);
			clVertex3((float)WINDOW_WIDTH, (float)WINDOW_HEIGHT, 0.0f);
			clVertex3(0.0f,				   (float)WINDOW_HEIGHT, 0.0f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();

		SetGLProgram(NULL);

		g_MVPMatrix->SetProjectionMode(CL_PROJECTION);
		g_MVPMatrix->PopMatrix();
	}

	GLProgram* DeferredLight_Directional::m_glProgram = NULL;
}
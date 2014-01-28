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

#include <nvToolsExt.h>

#include "DeferredLight_Point.h"
#include "Renderer/Texture/GLTexture.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Geometry/Sphere.h"

namespace CaptainLucha
{
	DeferredLight_Point::DeferredLight_Point()
		: Light(CL_POINT_LIGHT)
	{
		if(!m_glProgram)
		{
			m_glProgram = new GLProgram("Data/Shaders/DeferredVertShader.vert", "Data/Shaders/DeferredPointLightShader.frag");
			m_sphere = new Sphere(1.0f, 2);
		}
	}

	DeferredLight_Point::~DeferredLight_Point()
	{

	}

	void DeferredLight_Point::ApplyLight(
        const Vector3Df& cameraPos, 
        GLTexture* renderTarget0, 
        GLTexture* renderTarget1, 
        GLTexture* renderTarget2)
	{
        m_glProgram->SetUnifromTexture("renderTarget0", renderTarget0);
        m_glProgram->SetUnifromTexture("renderTarget1", renderTarget1);
        m_glProgram->SetUnifromTexture("renderTarget2", renderTarget2);

        m_glProgram->SetUniform("color", m_color);
        m_glProgram->SetUniform("camPos", cameraPos);
        m_glProgram->SetUniform("lightPos", m_position);
        m_glProgram->SetUniform("intensity", m_intensity);
        m_glProgram->SetUniform("radius", m_radius);

        glDisable(GL_DEPTH_TEST);
        glCullFace(GL_FRONT);

        DrawBSphere(*m_glProgram);

        glCullFace(GL_BACK);
        glEnable(GL_DEPTH_TEST);
	}
	void DeferredLight_Point::DrawBSphere(GLProgram& glProgram)
	{
		const Vector3Df& pos = GetPosition();
		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->LoadIdentity();
		g_MVPMatrix->Translate(pos);
        g_MVPMatrix->Scale(m_radius, m_radius, m_radius);
        m_sphere->Draw(glProgram);
		g_MVPMatrix->PopMatrix();
	}

	GLProgram* DeferredLight_Point::m_glProgram = NULL;
	GLProgram* DeferredLight_Point::m_nullProgram = NULL;
	Sphere* DeferredLight_Point::m_sphere = NULL;
}
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

#include "DeferredLight_Spot.h"
#include "Renderer/Texture/GLTexture.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Geometry/Cone.h"

namespace CaptainLucha
{
	DeferredLight_Spot::DeferredLight_Spot()
		: DeferredLight(CL_POINT_LIGHT)
	{
		if(!m_glProgram)
		{
			m_glProgram = new GLProgram("Data/Shaders/DeferredVertShader.vert", "Data/Shaders/DeferredSpotLight.frag");
		}
	}

	DeferredLight_Spot::~DeferredLight_Spot()
	{

	}

	void DeferredLight_Spot::ApplyLight(const Vector3Df& cameraPos, GLTexture* renderTarget0, GLTexture* renderTarget1, GLTexture* renderTarget2, GLTexture* renderTarget3)
	{
		UNUSED(renderTarget3)
		m_glProgram->SetUnifromTexture("renderTarget0", renderTarget0);
		m_glProgram->SetUnifromTexture("renderTarget1", renderTarget1);
		m_glProgram->SetUnifromTexture("renderTarget2", renderTarget2);

		m_glProgram->SetUniform("camPos", cameraPos);
		m_glProgram->SetUniform("color", m_color);
		m_glProgram->SetUniform("intensity", m_intensity);
		m_glProgram->SetUniform("radius", m_radius);

		m_glProgram->SetUniform("lightPos", (Vector3Df)GetPosition());
		m_glProgram->SetUniform("lightDir", m_lightDir);
		m_glProgram->SetUniform("innerAngle", cos(m_innerConeAngle));
		m_glProgram->SetUniform("outerAngle", cos(m_outerConeAngle));

		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->LoadIdentity();

		Matrix4Df rot;
		rot.MakeRotation(m_lightDir * -1, Vector3Df(0, 1, 0));

		g_MVPMatrix->Translate(GetPosition());
		g_MVPMatrix->MultMatrix(rot.GetTranspose());
		g_MVPMatrix->Translate(Vector3Df(0.0f, 0.0f, -m_radius));

		Cone cone((m_radius + 100.0f) * m_outerConeAngle, m_radius + 100.0f, 25, 25);
		cone.Draw(*m_glProgram);

		g_MVPMatrix->PopMatrix();
	}

	void DeferredLight_Spot::StencilPass()
	{

	}

	void DeferredLight_Spot::SetLookAt(const Vector3Df& lookAt)
	{
		m_lightDir = lookAt - GetPosition();
		m_lightDir.Normalize();
	}

	GLProgram* DeferredLight_Spot::m_glProgram = NULL;
}
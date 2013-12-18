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

#include "DeferredRenderer.h"

#include "Lights/DeferredLight_Point.h"
#include "Lights/DeferredLight_Ambient.h"
#include "Lights/DeferredLight_Directional.h"
#include "Lights/DeferredLight_Spot.h"

#include "RendererUtils.h"
#include "Renderer/FrameBufferObject.h"
#include <glew.h>
#include <iomanip>

namespace CaptainLucha
{
	DeferredRenderer::DeferredRenderer()
		: m_debugDraw(false)
	{
		InitRenderer();
	}

	DeferredRenderer::~DeferredRenderer()
	{

	}

	void DeferredRenderer::Draw()
	{
		PopulateGBuffers();

		BeginLightPass();
		LightPass();
		EndLightPass();
		RenderAccumulator();

		if(m_debugDraw)
			DebugRenderGBuffer();
	}

	void DeferredRenderer::AddObjectToRender(Object* object)
	{
		REQUIRES(object)
		m_renderableObjects.push_back(object);
	}

	void DeferredRenderer::RemoveObject(Object* object)
	{
		for(auto it = m_renderableObjects.begin(); it != m_renderableObjects.end(); ++it)
		{
			if((*it) == object)
			{
				m_renderableObjects.erase(it);
				return;
			}
		}
	}

	DeferredLight_Point* DeferredRenderer::CreateNewPointLight() 
	{
		DeferredLight_Point* newLight = new DeferredLight_Point();
		AddNewLight(newLight);
		return newLight;
	}

	DeferredLight_Ambient* DeferredRenderer::CreateNewAmbientLight()
	{
		DeferredLight_Ambient* newLight = new DeferredLight_Ambient();
		AddNewFullscreenLight(newLight);
		return newLight;
	}

	DeferredLight_Directional* DeferredRenderer::CreateNewDirectionalLight()
	{
		DeferredLight_Directional* newLight = new DeferredLight_Directional();
		AddNewFullscreenLight(newLight);
		return newLight;
	}

	DeferredLight_Spot* DeferredRenderer::CreateNewSpotLight()
	{
		DeferredLight_Spot* newLight = new DeferredLight_Spot();
		AddNewLight(newLight);
		return newLight;
	}

	int DeferredRenderer::GetObjectIDFromRT() const
	{
		int objectID[3];

		glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo0);
		glReadBuffer(GL_COLOR_ATTACHMENT3);

		glReadPixels(WINDOW_WIDTH >> 1, WINDOW_HEIGHT >> 1, 1, 1, GL_RGB_INTEGER, GL_UNSIGNED_INT, &objectID);

		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		return objectID[0];
	}

	void DeferredRenderer::RemoveLight(DeferredLight* light)
	{
		REQUIRES(light)

		for(size_t i = 0; i < m_lights.size(); ++i)
		{
			if(light == m_lights[i])
			{
				delete m_lights[i];
				m_lights[i] = NULL;
				return;
			}
		}

		for(size_t i = 0; i < m_fullscreenLights.size(); ++i)
		{
			if(light == m_fullscreenLights[i])
			{
				delete m_fullscreenLights[i];
				m_fullscreenLights[i] = NULL;
				return;
			}
		}
	}

	void DeferredRenderer::AddNewLight(DeferredLight* newLight)
	{
		REQUIRES(newLight)

		for(size_t i = 0; i < m_lights.size(); ++i)
		{
			if(m_lights[i] == NULL)
			{
				m_lights[i] = newLight;
				return;
			}
		}

		m_lights.push_back(newLight);
	}

	void DeferredRenderer::AddNewFullscreenLight(DeferredLight* newLight)
	{
		REQUIRES(newLight)

		for(size_t i = 0; i < m_fullscreenLights.size(); ++i)
		{
			if(m_fullscreenLights[i] == NULL)
			{
				m_fullscreenLights[i] = newLight;
				return;
			}
		}

		m_fullscreenLights.push_back(newLight);
	}

	void DeferredRenderer::DrawScene(GLProgram& program)
	{		
		for(size_t i = 0; i < m_renderableObjects.size(); ++i)
		{
			if(m_renderableObjects[i]->IsVisible())
			{
				program.SetUniform("ObjectID", m_renderableObjects[i]->GetID());
				m_renderableObjects[i]->Draw(program);
			}
		}
	}

	void DeferredRenderer::PopulateGBuffers()
	{
		glBlendFunc(GL_ONE, GL_ZERO);

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo0);
		{
			ClearFBO();

			GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
			glDrawBuffers(4, buffers);

			g_MVPMatrix->SetViewMatrix(m_viewMatrix);
			DrawScene(*m_graphicsBufferProgram);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void DeferredRenderer::RenderAccumulator()
	{
		g_MVPMatrix->SetProjectionMode(CL_ORTHOGRAPHIC);

		SetGLProgram(m_finalPassProgram);

		SetTexture("renderTarget0", m_rt0);
		SetTexture("renderTarget1", m_rt1);
		SetTexture("renderTarget2", m_rt2);
		SetTexture("renderTarget3", m_rt3);
		SetTexture("depth", m_depth);
		SetTexture("accumulatorDiffuse", m_accumDiffuseLightTexture);
		SetTexture("accumulatorSpecular", m_accumSpecularLightTexture);

		FullScreenPass();

		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

		SetGLProgram(NULL);

		g_MVPMatrix->SetProjectionMode(CL_PROJECTION);
	}

	void DeferredRenderer::LightPass()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo1);
		{
			GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
			glDrawBuffers(2, buffers);

			ClearFBO();

			glEnable(GL_STENCIL_TEST);
			for(size_t i = 0; i < m_lights.size(); ++i)
			{
				m_lights[i]->StencilPass();
				m_lights[i]->ApplyLight(m_cameraPos, m_rt0, m_rt1, m_rt2, m_rt3);
			}
			glDisable(GL_STENCIL_TEST);

			for(size_t i = 0; i < m_fullscreenLights.size(); ++i)
			{
				m_fullscreenLights[i]->ApplyLight(m_cameraPos, m_rt0, m_rt1, m_rt2, m_rt3);
			}
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void DeferredRenderer::BeginLightPass()
	{
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);

		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
	}
	
	void DeferredRenderer::EndLightPass()
	{
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_STENCIL_TEST);
	}

	bool DeferredRenderer::ValidateFBO()
	{
		GLenum status;

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo0);
		status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			traceWithFile("Frame buffer ERROR!");
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return false;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo1);
		status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			traceWithFile("Frame buffer ERROR!");
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return false;
		}

		return true;
	}

	void DeferredRenderer::ClearFBO()
	{
		glClearDepth( 1.f );
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	}

	void DeferredRenderer::InitRenderer()
	{
		//////////////////////////////////////////////////////////////////////////
		//FrameBuffer GBuffer
		unsigned int rt0 = 0;
		unsigned int rt1 = 0;
		unsigned int rt2 = 0;
		unsigned int rt3 = 0;
		unsigned int depth = 0;

		//Diffuse Target
		//
		glGenTextures(1, &rt0);
		glBindTexture(GL_TEXTURE_2D, rt0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT,
			0,	GL_RGBA, GL_FLOAT, NULL);

		//Normal Target
		//
		glGenTextures(1, &rt1);
		glBindTexture(GL_TEXTURE_2D, rt1);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT,
			0,	GL_RGBA, GL_FLOAT, NULL);

		//World Position Target
		//
		glGenTextures(1, &rt2);
		glBindTexture(GL_TEXTURE_2D, rt2);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT,
			0,	GL_RGBA, GL_FLOAT, NULL);

		//Object IDs
		//
		glGenTextures(1, &rt3);
		glBindTexture(GL_TEXTURE_2D, rt3);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16UI, WINDOW_WIDTH, WINDOW_HEIGHT,
			0,	GL_RGBA_INTEGER, GL_UNSIGNED_INT, NULL);

		glGenTextures(1, &depth);
		glBindTexture(GL_TEXTURE_2D, depth);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH32F_STENCIL8, WINDOW_WIDTH, WINDOW_HEIGHT,
			0,	GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		glGenFramebuffers(1, &m_fbo0); 
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo0);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt0, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rt1, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, rt2, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, rt3, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

		m_rt0 = new GLTexture(rt0, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_rt1 = new GLTexture(rt1, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_rt2 = new GLTexture(rt2, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_rt3 = new GLTexture(rt3, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_depth = new GLTexture(depth, WINDOW_WIDTH, WINDOW_HEIGHT);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		//////////////////////////////////////////////////////////////////////////
		//FrameBuffer Accumulator
		unsigned int color1, color2;

		glGenTextures(1, &color1);
		glBindTexture(GL_TEXTURE_2D, color1);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT,
			0,	GL_RGBA, GL_FLOAT, NULL);

		glGenTextures(1, &color2);
		glBindTexture(GL_TEXTURE_2D, color2);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT,
			0,	GL_RGBA, GL_FLOAT, NULL);

		glGenFramebuffers(1, &m_fbo1);
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo1);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color1, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, color2, 0);
		
		m_accumDiffuseLightTexture = new GLTexture(color1, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_accumSpecularLightTexture = new GLTexture(color2, WINDOW_WIDTH, WINDOW_HEIGHT);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		m_graphicsBufferProgram = new GLProgram("Data/Shaders/DeferredGraphicsShader.vert", "Data/Shaders/DeferredGraphicsShader.frag");
		m_debugDepthProgram = new GLProgram("Data/Shaders/SimpleShader.vert", "Data/Shaders/DepthDebug.frag");
		m_finalPassProgram = new GLProgram("Data/Shaders/SimpleShader.vert", "Data/Shaders/DeferredFinalPass.frag");
	
		ValidateFBO();
	}

	void DeferredRenderer::FullScreenPass()
	{
		g_MVPMatrix->SetProjectionMode(CL_ORTHOGRAPHIC);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.0f);
			clVertex3(WINDOW_WIDTHf, 0.0f, 0.0f);
			clVertex3(WINDOW_WIDTHf, WINDOW_HEIGHTf, 0.0f);
			clVertex3(0.0f, WINDOW_HEIGHTf, 0.0f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();
		g_MVPMatrix->SetProjectionMode(CL_PROJECTION);
	}

	void DeferredRenderer::DebugRenderGBuffer()
	{
		g_MVPMatrix->SetProjectionMode(CL_ORTHOGRAPHIC);
		g_MVPMatrix->PushMatrix();

		std::stringstream ss;
		ss << " Diffuse              Normals             World Pos             Emissive             Light Accum           Spec Accum           Depth Buffer";
		Draw2DDebugText(Vector2Df(0.0f, WINDOW_HEIGHT / 10.0f), ss.str().c_str());

		SetUniform("useAlpha", false);
		SetTexture("diffuseMap", m_rt0);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, 0.1f, 0.0f);
			clVertex3(WINDOW_WIDTH / 10.0f, WINDOW_HEIGHT / 10.0f, 0.1f);
			clVertex3(0.0f, WINDOW_HEIGHT / 10.0f, 0.1f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();

		g_MVPMatrix->Translate(WINDOW_WIDTH / 10.0f, 0.0f, 0.0f);
		SetTexture("diffuseMap", m_rt1);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, WINDOW_HEIGHT / 10.0f, 0.1f);
			clVertex3(0.0f, WINDOW_HEIGHT / 10.0f, 0.1f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();

		g_MVPMatrix->Translate(WINDOW_WIDTH / 10.0f, 0.0f, 0.0f);
		SetTexture("diffuseMap", m_rt2);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, WINDOW_HEIGHT / 10.0f, 0.1f);
			clVertex3(0.0f, WINDOW_HEIGHT / 10.0f, 0.1f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();

		g_MVPMatrix->Translate(WINDOW_WIDTH / 10.0f, 0.0f, 0.0f);
		SetTexture("diffuseMap", m_rt3);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, WINDOW_HEIGHT / 10.0f, 0.1f);
			clVertex3(0.0f, WINDOW_HEIGHT / 10.0f, 0.1f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();

		g_MVPMatrix->Translate(WINDOW_WIDTH / 10.0f, 0.0f, 0.0f);
		SetTexture("diffuseMap", m_accumDiffuseLightTexture);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, WINDOW_HEIGHT / 10.0f, 0.1f);
			clVertex3(0.0f, WINDOW_HEIGHT / 10.0f, 0.1f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();

		g_MVPMatrix->Translate(WINDOW_WIDTH / 10.0f, 0.0f, 0.0f);
		SetTexture("diffuseMap", m_accumSpecularLightTexture);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, WINDOW_HEIGHT / 10.0f, 0.1f);
			clVertex3(0.0f, WINDOW_HEIGHT / 10.0f, 0.1f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();

		g_MVPMatrix->Translate(WINDOW_WIDTH / 10.0f, 0.0f, 0.0f);
		SetGLProgram(m_debugDepthProgram);
		SetTexture("depthTexture", m_depth);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, 0.0f, 0.1f);
			clVertex3(WINDOW_WIDTH / 10.0f, WINDOW_HEIGHT / 10.0f, 0.1f);
			clVertex3(0.0f, WINDOW_HEIGHT / 10.0f, 0.1f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();
		SetGLProgram(NULL);

		SetUniform("useAlpha", true);

		g_MVPMatrix->PopMatrix();
		g_MVPMatrix->SetProjectionMode(CL_PROJECTION);

	}
}
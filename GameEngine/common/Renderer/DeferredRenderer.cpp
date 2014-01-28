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
#include "Renderable.h"

#include "Lights/DeferredLight_Point.h"
#include "Lights/DeferredLight_Ambient.h"
#include "Lights/DeferredLight_Directional.h"
#include "Lights/DeferredLight_Spot.h"
#include "Time/ProfilingSection.h"

#include "RendererUtils.h"

#include "Renderer/FrameBufferObject.h"
#include <glew.h>
#include <iomanip>
#include <nvToolsExt.h>

namespace CaptainLucha
{
	DeferredRenderer::DeferredRenderer()
        : m_deferredFBO(0)
	{
		DeferredRenderer::InitRenderer();
	}

	DeferredRenderer::~DeferredRenderer()
	{

	}

	void DeferredRenderer::Draw()
	{
        ClearColorDepth();

        g_MVPMatrix->UpdateFrustum(m_cameraPos, m_cameraDir);

        if(HasReflectiveObject())
            ReflectivePass();

        DrawPass();
	}

	Light* DeferredRenderer::CreateNewPointLight() 
	{
		Light* newLight = new DeferredLight_Point();
		AddNewLight(newLight);
		return newLight;
	}

	Light* DeferredRenderer::CreateNewAmbientLight()
	{
		Light* newLight = new DeferredLight_Ambient();
		AddNewFullscreenLight(newLight);
		return newLight;
	}

	Light_Directional* DeferredRenderer::CreateNewDirectionalLight()
	{
		DeferredLight_Directional* newLight = new DeferredLight_Directional();
		AddNewFullscreenLight(newLight);
		return newLight;
	}

	Light_Spot* DeferredRenderer::CreateNewSpotLight()
	{
		DeferredLight_Spot* newLight = new DeferredLight_Spot();
		AddNewLight(newLight);
		return newLight;
	}

	void DeferredRenderer::RemoveLight(Light* light)
	{
		REQUIRES(light)

		for(size_t i = 0; i < m_deferredLights.size(); ++i)
		{
			if(light == m_deferredLights[i])
			{
				delete m_deferredLights[i];
				m_deferredLights[i] = NULL;
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

	void DeferredRenderer::AddNewLight(Light* newLight)
	{
		REQUIRES(newLight)

		for(size_t i = 0; i < m_deferredLights.size(); ++i)
		{
			if(m_deferredLights[i] == NULL)
			{
				m_deferredLights[i] = newLight;
				return;
			}
		}

		m_deferredLights.push_back(newLight);
	}

	void DeferredRenderer::AddNewFullscreenLight(Light* newLight)
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

    void DeferredRenderer::ReflectivePass()
    {
        m_currentPass = CL_REFLECT_PASS;

        g_MVPMatrix->PushMatrix();
        g_MVPMatrix->Translate(m_reflectivePos * 2.0f);
        g_MVPMatrix->Scale(1.0f, -1.0f, 1.0f); //todo reflect base on surface normal

        glCullFace(GL_FRONT);
        PopulateGBuffers();
        glCullFace(GL_BACK);

        DeferredLightPass();

        FinalPass();

        glBindFramebuffer(GL_FRAMEBUFFER, m_reflectiveFBO);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        RenderToScreen(m_worldPosRT, m_reflectiveWidth, m_reflectiveHeight);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        g_MVPMatrix->PopMatrix();
    }

    void DeferredRenderer::DrawPass()
    {
        m_currentPass = CL_DRAW_PASS;

        PopulateGBuffers();
        DeferredLightPass();
        FinalPass();

        if(m_reflectiveObject)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, m_deferredFBO);
            glDrawBuffer(GL_COLOR_ATTACHMENT0);
            glClear(GL_COLOR_BUFFER_BIT);
            glDepthMask(GL_FALSE);
            RenderToScreen(m_worldPosRT);
            glDepthMask(GL_TRUE);
            m_reflectiveObject->Draw(*m_graphicsBufferProgram, *this);
            glDrawBuffer(GL_COLOR_ATTACHMENT2);
            RenderToScreen(m_colorRT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        RenderToScreen(m_worldPosRT);
    }

	void DeferredRenderer::PopulateGBuffers()
	{
		glBlendFunc(GL_ONE, GL_ZERO);
		glBindFramebuffer(GL_FRAMEBUFFER, m_deferredFBO);
		{
			GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
			glDrawBuffers(3, buffers);

            ClearColorDepth();

			g_MVPMatrix->SetViewMatrix(m_viewMatrix);
			DrawScene(*m_graphicsBufferProgram, false);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

    void DeferredRenderer::RenderAccumulator()
    {
		g_MVPMatrix->SetProjectionMode(CL_ORTHOGRAPHIC);

		SetGLProgram(m_finalPassProgram);

		SetTexture("renderTarget0", m_colorRT);
        SetTexture("renderTarget1", m_normalRT);
		SetTexture("accumulatorDiffuse", m_accumDiffuseLightTexture);
		SetTexture("accumulatorSpecular", m_accumSpecularLightTexture);

		FullScreenPass();
		SetGLProgram(NULL);

		g_MVPMatrix->SetProjectionMode(CL_PROJECTION);
	}

	void DeferredRenderer::DeferredLightPass()
	{
        glDepthMask(GL_FALSE);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);

		glBindFramebuffer(GL_FRAMEBUFFER, m_deferredFBO);
		{
			GLenum buffers[] = {GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4};
			glDrawBuffers(2, buffers);

            glClear(GL_COLOR_BUFFER_BIT);

			for(size_t i = 0; i < m_deferredLights.size(); ++i)
			{
                if(g_MVPMatrix->GetViewFrustum().SphereInFrustum(m_deferredLights[i]->GetPosition(), m_deferredLights[i]->GetRadius()))
				    m_deferredLights[i]->ApplyLight(m_cameraPos, m_colorRT, m_normalRT, m_worldPosRT);
			}

			for(size_t i = 0; i < m_fullscreenLights.size(); ++i)
			{
				m_fullscreenLights[i]->ApplyLight(m_cameraPos, m_colorRT, m_normalRT, m_worldPosRT);
			}
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

    void DeferredRenderer::FinalPass()
    {
        //Do forward pass for skybox
        glDepthMask(GL_FALSE);
        glBindFramebuffer(GL_FRAMEBUFFER, m_deferredFBO);
        glDrawBuffer(GL_COLOR_ATTACHMENT2);
        RenderAccumulator();
        DrawSkybox();
        DrawScene(*m_graphicsBufferProgram, true);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDepthMask(GL_TRUE);
    }

    void DeferredRenderer::RenderToScreen(
        GLTexture* diffuseTex,
        int screenWidth, 
        int screenHeight)
    {
        //Draw final scene
        SetGLProgram(NULL);
        SetUtilsColor(Color::White);
        SetUniform("useTexture", true);
        SetTexture("diffuseMap", diffuseTex);
        FullScreenPass(screenWidth, screenHeight);
    }

	bool DeferredRenderer::ValidateFBO()
	{
		GLenum status;

		glBindFramebuffer(GL_FRAMEBUFFER, m_deferredFBO);
		status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			traceWithFile("Frame buffer ERROR!");
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return false;
		}
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

		return true;
	}

	void DeferredRenderer::ClearColorDepth()
	{
		glClearDepth( 1.f );
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void DeferredRenderer::InitRenderer()
	{
		//////////////////////////////////////////////////////////////////////////
		//FrameBuffer GBuffer
		unsigned int rt0 = 0;
		unsigned int rt1 = 0;
		unsigned int rt2 = 0;
		unsigned int depth = 0;
        unsigned int color1 = 0;
        unsigned int color2 = 0;

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

		glGenTextures(1, &depth);
		glBindTexture(GL_TEXTURE_2D, depth);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT,
			0,	GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		glGenFramebuffers(1, &m_deferredFBO); 
		glBindFramebuffer(GL_FRAMEBUFFER, m_deferredFBO);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt0,    0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rt1,    0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, rt2,    0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, color1, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, color2, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, depth,  0);

		m_colorRT = new GLTexture(rt0, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_normalRT = new GLTexture(rt1, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_worldPosRT = new GLTexture(rt2, WINDOW_WIDTH, WINDOW_HEIGHT);
        m_finalRenderSceneTex = m_worldPosRT;
        m_accumDiffuseLightTexture = new GLTexture(color1, WINDOW_WIDTH, WINDOW_HEIGHT);
        m_accumSpecularLightTexture = new GLTexture(color2, WINDOW_WIDTH, WINDOW_HEIGHT);
		m_depth = new GLTexture(depth, WINDOW_WIDTH, WINDOW_HEIGHT);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		m_graphicsBufferProgram = new GLProgram("Data/Shaders/DeferredGraphicsShader.vert", "Data/Shaders/DeferredGraphicsShader.frag");
		m_finalPassProgram = new GLProgram("Data/Shaders/SimpleShader.vert", "Data/Shaders/DeferredFinalPass.frag");
	
		ValidateFBO();
	}
}
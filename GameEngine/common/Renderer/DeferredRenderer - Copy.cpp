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
#include "Time/ProfilingSection.h"

#include "RendererUtils.h"

#include "Renderer/FrameBufferObject.h"
#include <glew.h>
#include <iomanip>
#include <nvToolsExt.h>

namespace CaptainLucha
{
    static const unsigned int GBUFFER_DRAW_STENCIL_MASK = 0xF0;
    static const unsigned int GBUFFER_LIGHT_STENCIL_MASK = 0x0F;
    static const unsigned int GBUFFER_DRAW_STENCIL_REF  = 0xF0;

	DeferredRenderer::DeferredRenderer()
	{
		InitRenderer();
	}

	DeferredRenderer::~DeferredRenderer()
	{

	}

	void DeferredRenderer::Draw()
	{
        g_MVPMatrix->SetViewMatrix(m_viewMatrix);
        g_MVPMatrix->UpdateFrustum(m_cameraPos, m_cameraDir);

        //clear main buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        ClearGBuffer();

        glEnable(GL_STENCIL_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, m_GBufferFBO);
        for(int i = 0; i < 1; ++i)
        {
            glStencilMask(0xFF);
            glClear(GL_STENCIL_BUFFER_BIT);
            glStencilFunc(GL_ALWAYS, GBUFFER_DRAW_STENCIL_REF, GBUFFER_DRAW_STENCIL_MASK);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            PopulateGBuffers(i != 0);

            ClearLightAccumBufferWithStencil();

            LightPass();
            
            BlendBackgroundTextureToAccum();
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_STENCIL_TEST);

        glDisable(GL_DEPTH_TEST);
        SetGLProgram(NULL);
        SetTexture("diffuseMap", m_accumDiffuseLightTexture);
        SetUtilsColor(Color::White);
        SetUniform("useTexture", true);
        FullScreenPass();
        glEnable(GL_DEPTH_TEST);

        //RenderAccumulator();
	}

	Light* DeferredRenderer::CreateNewPointLight() 
	{
		DeferredLight_Point* newLight = new DeferredLight_Point();
		AddNewLight(newLight);
		return newLight;
	}

	Light* DeferredRenderer::CreateNewAmbientLight()
	{
		DeferredLight_Ambient* newLight = new DeferredLight_Ambient();
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

	void DeferredRenderer::PopulateGBuffers(bool isAlphaPass)
	{
        m_graphicsBufferProgram->SetUnifromTexture("lightAccum", m_accumDiffuseLightTexture);

		glBlendFunc(GL_ONE, GL_ZERO);

        GLenum buffers[] = {
            GL_COLOR_ATTACHMENT0, 
            GL_COLOR_ATTACHMENT1, 
            GL_COLOR_ATTACHMENT2};
        glDrawBuffers(3, buffers);

        if(isAlphaPass)
            DrawSceneAlphaPass(*m_graphicsBufferProgram);
        else
            DrawScene(*m_graphicsBufferProgram);

        m_graphicsBufferProgram->SetUnifromTexture("lightAccum", NULL);
	}

    void DeferredRenderer::BlendBackgroundTextureToAccum()
    {
        SetGLProgram(NULL);
        SetUtilsColor(Color::White);
        SetUniform("useTexture", true);

        glDisable(GL_DEPTH_TEST);
        GLenum buffers[] = {GL_COLOR_ATTACHMENT3};
        glDrawBuffers(1, buffers);

        //Multiply background to diffuse + spec LA
        glBlendFunc(GL_DST_COLOR, GL_ZERO);
        SetTexture("diffuseMap", m_rtColor);
        FullScreenPass();

        glEnable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE, GL_ONE);
    }

	void DeferredRenderer::LightPass()
	{
        glDepthMask(GL_FALSE);
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE, GL_ONE);

        GLenum buffers[] = {GL_COLOR_ATTACHMENT3};
        glDrawBuffers(2, buffers);

        for(size_t i = 0; i < m_deferredLights.size(); ++i)
        {
            //doesn't work with spotlight dur
            if(g_MVPMatrix->GetViewFrustum().SphereInFrustum(
                m_deferredLights[i]->GetPosition(),
                m_deferredLights[i]->GetRadius()))
            {
                //m_deferredLights[i]->StencilPass();

                //glStencilFunc(GL_ALWAYS, GBUFFER_DRAW_STENCIL_REF, GBUFFER_DRAW_STENCIL_MASK);
                //glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

                m_deferredLights[i]->ApplyLight(
                    m_cameraPos, 
                    m_rtColor, 
                    m_rtNormalEmissive, 
                    m_rtPosSpecIntensity);
            }
        }

        for(size_t i = 0; i < m_fullscreenLights.size(); ++i)
        {
            m_fullscreenLights[i]->ApplyLight(
                m_cameraPos, 
                m_rtColor,
                m_rtNormalEmissive,
                m_rtPosSpecIntensity);
        }

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	bool DeferredRenderer::ValidateFBO(unsigned int fbo)
	{
		GLenum status;

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			traceWithFile("Frame buffer ERROR!");
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return false;
		}

		return true;
	}

    void DeferredRenderer::ClearGBuffer()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, m_GBufferFBO);
        GLenum buffers[] = {
            GL_COLOR_ATTACHMENT0, 
            GL_COLOR_ATTACHMENT1, 
            GL_COLOR_ATTACHMENT2, 
            GL_COLOR_ATTACHMENT3};
        glDrawBuffers(4, buffers);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void DeferredRenderer::ClearLightAccumBufferWithStencil()
    {
        glDisable(GL_DEPTH_TEST);

        glStencilFunc(GL_EQUAL, GBUFFER_DRAW_STENCIL_REF, GBUFFER_DRAW_STENCIL_MASK);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

        GLenum buffers[] = {GL_COLOR_ATTACHMENT3};
        glDrawBuffers(1, buffers);
        SetUtilsColor(Color::Black);
        FullScreenPass();

        glEnable(GL_DEPTH_TEST);
    }

	void DeferredRenderer::InitRenderer()
	{
        CreateGBufferFBO();

		m_graphicsBufferProgram = new GLProgram("Data/Shaders/DeferredGraphicsShader.vert", "Data/Shaders/DeferredGraphicsShader.frag");
		m_debugDepthProgram = new GLProgram("Data/Shaders/SimpleShader.vert", "Data/Shaders/DepthDebug.frag");
		m_finalPassProgram = new GLProgram("Data/Shaders/SimpleShader.vert", "Data/Shaders/DeferredFinalPass.frag");
	}

    void DeferredRenderer::CreateGBufferFBO()
    {
        unsigned int rt0 = 0;
        unsigned int rt1 = 0;
        unsigned int rt2 = 0;
        unsigned int color1 = 0;
        unsigned int depth = 0;

        //////////////////////////////////////////////////////////////////////////
        //  Create rendertarget textures
        //////////////////////////////////////////////////////////////////////////
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

        //Light Accum
        //
        glGenTextures(1, &color1);
        glBindTexture(GL_TEXTURE_2D, color1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT,
            0,	GL_RGBA, GL_FLOAT, NULL);

        //Depth
        //
        glGenTextures(1, &depth);
        glBindTexture(GL_TEXTURE_2D, depth);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH32F_STENCIL8, WINDOW_WIDTH, WINDOW_HEIGHT,
            0,	GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

        //////////////////////////////////////////////////////////////////////////
        //  Create FBO
        //////////////////////////////////////////////////////////////////////////
        glGenFramebuffers(1, &m_GBufferFBO);    
        glBindFramebuffer(GL_FRAMEBUFFER, m_GBufferFBO);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt0, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rt1, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, rt2, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, color1, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

        m_rtColor = new GLTexture(rt0, WINDOW_WIDTH, WINDOW_HEIGHT);
        m_rtNormalEmissive = new GLTexture(rt1, WINDOW_WIDTH, WINDOW_HEIGHT);
        m_rtPosSpecIntensity = new GLTexture(rt2, WINDOW_WIDTH, WINDOW_HEIGHT);
        m_accumDiffuseLightTexture = new GLTexture(color1, WINDOW_WIDTH, WINDOW_HEIGHT);
        m_depth = new GLTexture(depth, WINDOW_WIDTH, WINDOW_HEIGHT);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        ValidateFBO(m_GBufferFBO);
    }

}
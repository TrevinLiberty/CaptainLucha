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

#include "FrameBufferObject.h"

#include "RendererUtils.h"
#include "Utils/UtilDebug.h"

#include <glew.h>

namespace CaptainLucha
{
	FrameBufferObject::FrameBufferObject(int textureWidth, int textureHeight, bool depthOnly)
		: m_clearColor(Color::White),
		  m_color(NULL),
		  m_depth(NULL)
	{
		//Gen Frame Buffer
		//
		glGenFramebuffers(1, &m_fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

		unsigned int textureColor_ = 0;
		unsigned int textureDepth_ = 0;

		if(!depthOnly)
		{
			//Gen Textures
			//
			glGenTextures(1, &textureColor_);
			glBindTexture(GL_TEXTURE_2D, textureColor_);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	
			glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, textureWidth, textureHeight,
				0,	GL_RGB, GL_UNSIGNED_BYTE, NULL);

			m_color = new GLTexture(textureColor_, textureWidth, textureHeight);
		}
		
		glGenTextures(1, &textureDepth_);
		glBindTexture(GL_TEXTURE_2D, textureDepth_);

		float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, textureWidth, textureHeight,
			0,	GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		m_depth = new GLTexture(textureDepth_, textureWidth, textureHeight);

		glBindTexture(GL_TEXTURE_2D, 0);

		if(!depthOnly)
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColor_, 0);
		
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, textureDepth_, 0);

		CheckFBO();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	FrameBufferObject::~FrameBufferObject()
	{
		glDeleteFramebuffers(1, &m_fbo);
	}

	void FrameBufferObject::Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
	}

	void FrameBufferObject::UnBind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void FrameBufferObject::DrawBuffer(bool colorBuffer)
	{
		SetColor(Vector4Df(1.0f, 1.0f, 1.0f, 1.0f));

		if(colorBuffer)
			SetTexture("diffuseMap", m_color);
		else
			SetTexture("diffuseMap", m_depth);

		DrawBegin(CL_QUADS);
		clVertex3(0.0f, 0.0f, 0.0f);
		clVertex3(WINDOW_WIDTH / 10.0f, 0.0f, 0.0f);
		clVertex3(WINDOW_WIDTH / 10.0f, WINDOW_HEIGHT / 10.0f, 0.0f);
		clVertex3(0.0f, WINDOW_HEIGHT / 10.0f, 0.0f);

		clTexCoord(0, 0);
		clTexCoord(1, 0);
		clTexCoord(1, 1);
		clTexCoord(0, 1);
		DrawEnd();
	}

	void FrameBufferObject::ClearBuffer()
	{
		glClearDepth( 1.f );
		glClearColor(m_clearColor.r, m_clearColor.g, m_clearColor.b, m_clearColor.a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_CULL_FACE);
	}

	bool FrameBufferObject::CheckFBO() const
	{
		GLenum status;

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			traceWithFile("Frame buffer ERROR!");
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return false;
		}

		return true;
	}
}

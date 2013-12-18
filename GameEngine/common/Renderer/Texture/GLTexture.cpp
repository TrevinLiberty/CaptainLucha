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

#include "GLTexture.h"
#include "Utils/CLFileImporter.h"

#include <FreeImage.h>
#include <FreeImagePlus.h>
#include <glew.h>
#include <GL/glu.h>

namespace CaptainLucha
{
	GLTexture::GLTexture()
		: m_glTextureID(0),
		  m_width(0),
		  m_height(0),
		  m_isValid(false),
		  m_inFactory(false)
	{}

	GLTexture::GLTexture(unsigned int textureID, int width, int height)
		: m_glTextureID(textureID),
		m_width(width),
		m_height(height),
		m_isValid(true),
		m_inFactory(false)
	{

	}

	GLTexture::GLTexture(const GLTexture& rhs)
	{
		*this = rhs;
	}

	GLTexture::GLTexture(const char* texturePath, bool genMipMaps)
		: m_glTextureID(0),
		  m_width(0),
		  m_height(0),
		  m_isValid(false),
		  m_inFactory(false)
	{
		LoadFromFile(texturePath, genMipMaps);
	}

	void GLTexture::operator=(const GLTexture& rhs)
	{
		m_glTextureID = rhs.m_glTextureID;
		m_width = rhs.m_width;
		m_height = rhs.m_height;
		m_isValid = rhs.m_isValid;
		m_inFactory = rhs.m_inFactory;
	}

	GLTexture::~GLTexture()
	{
		
	}

	void GLTexture::LoadFromFile(const char* texturePath, bool createMipMap)
	{
		int width = 0;
		int height = 0;

		CLFileImporter fileImporter;
		CLFile* file = fileImporter.LoadFile(texturePath);

		if(file != NULL)
		{
			fipMemoryIO memIO((BYTE*)file->m_buffer, file->m_bufLength);

			if(memIO.isValid())
			{
				BYTE* texels = 0;
				FIBITMAP* bitmap = 0;

				FREE_IMAGE_FORMAT format = FIF_UNKNOWN;
				format = memIO.getFileType();

				if(format == FIF_UNKNOWN)
					format = FreeImage_GetFIFFromFilename(texturePath);
				if(format == FIF_UNKNOWN) 
					BadFormatError(texturePath);

				if(FreeImage_FIFSupportsReading(format))
					bitmap = memIO.load(format);
				else
					UnSupportedFormatError(texturePath);

				if(!bitmap)
					LoadFromFileError(texturePath);

				bitmap = FreeImage_ConvertTo32Bits(bitmap);
				texels = FreeImage_GetBits(bitmap);
				width = FreeImage_GetWidth(bitmap);
				height = FreeImage_GetHeight(bitmap);

				if(texels)
				{
					if(!createMipMap)
						LoadFromMemory(width, height, (unsigned int*)texels);
					else
						LoadFromMemoryMipMap(width, height, (unsigned int*)texels);

					FreeImage_Unload(bitmap);
					delete file;
					m_isValid = true;
					return;
				}
				else
					LoadFromFileError(texturePath);
			}
		}
		else
			LoadFromFileError(texturePath);
	}

	void GLTexture::SetMinMagFilterNearest()
	{
		glBindTexture(GL_TEXTURE_2D, m_glTextureID);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void GLTexture::SetMinMagFilterLinear()
	{
		glBindTexture(GL_TEXTURE_2D, m_glTextureID);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	//Does nothing if texture was obtained through the TextureFactoryGL
	void GLTexture::DeleteTexture()
	{
		if(!m_inFactory)
		{
			glDeleteTextures(1, &m_glTextureID);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void GLTexture::LoadFromMemory(int width, int height, unsigned int* texels)
	{
		m_width = width;
		m_height = height;
		glGenTextures(1, &m_glTextureID);
		glBindTexture(GL_TEXTURE_2D, m_glTextureID);

		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height,
			0,	GL_BGRA, GL_UNSIGNED_BYTE, texels);
	}

	void GLTexture::LoadFromMemoryMipMap(int width, int height, unsigned int* texels)
	{
		m_width = width;
		m_height = height;
		glGenTextures(1, &m_glTextureID);
		glBindTexture(GL_TEXTURE_2D, m_glTextureID);

		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, width, height, GL_BGRA, GL_UNSIGNED_BYTE, texels);
	}

	void GLTexture::LoadFromFileError(const char* texturePath)
	{
		std::stringstream ss;
		ss << "Unable to Load Texture:\n" << texturePath;
		MessageBox(NULL, ss.str().c_str(), "TEXTURE LOAD ERROR", MB_OK | MB_ICONERROR);
	}

	void GLTexture::BadFormatError(const char* texturePath)
	{
		std::stringstream ss;
		ss << "Unable to Load Texture:\n" << texturePath;
		MessageBox(NULL, "BAD TEXTURE FORMAT", ss.str().c_str(), MB_OK | MB_ICONERROR);
	}

	void GLTexture::UnSupportedFormatError(const char* texturePath)
	{
		std::stringstream ss;
		ss << "Unable to Load Texture:\n" << texturePath;
		MessageBox(NULL, "UNSUPPORTED TEXTURE FORMAT", ss.str().c_str(), MB_OK | MB_ICONERROR);
	}
}
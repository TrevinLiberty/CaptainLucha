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
 *	@file	GLTexture.h
 *	@brief	
 *
/****************************************************************************/

#ifndef TEXTUREGL_H_CL
#define TEXTUREGL_H_CL

#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class GLTexture
	{
	public:
		GLTexture();
		GLTexture(unsigned int textureID, int width, int height);
		GLTexture(const GLTexture& rhs);
		GLTexture(const char* texturePath, bool genMipMaps = false);
		~GLTexture();

		void operator=(const GLTexture& rhs);

		unsigned int GetTextureID() const {return m_glTextureID;}

		void LoadFromFile(const char* texturePath, bool createMipMap = false);

		unsigned int GetWidth() const {return m_width;}
		unsigned int GetHeight() const {return m_height;}

		bool IsValid() const {return m_isValid;}

		void SetMinMagFilterNearest();
		void SetMinMagFilterLinear();

		//Does nothing if texture was obtained through the TextureFactoryGL
		void DeleteTexture();

	protected:
		inline bool IsPowerOfTwo(int val) const {return (val % 2) == 0;}

		void LoadFromMemory(int width, int height, unsigned int* texels);
		void LoadFromMemoryMipMap(int width, int height, unsigned int* texels);

		void LoadFromFileError(const char* texturePath);
		void BadFormatError(const char* texturePath);
		void UnSupportedFormatError(const char* texturePath);

	private:
		unsigned int m_glTextureID;
		unsigned int m_width;
		unsigned int m_height;

		bool m_isValid;
		bool m_inFactory;

		friend class GLTextureManager;
	};
}

#endif
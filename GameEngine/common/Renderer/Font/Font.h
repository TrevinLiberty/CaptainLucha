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
 *	@file	Font.h
 *	@brief	
 *
/****************************************************************************/

#ifndef FONT_H_CL
#define FONT_H_CL

#include "Math/Vector2D.h"
#include "Math/Vector3D.h"
#include "Renderer/Color.h"
#include "freetype-gl/freetype-gl.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	struct CLFile;
	class GLTexture;
	class GLProgram;

	class Font
	{
	public:
		Font(const char* filepath, int size);
		~Font();

		//Very slow atm.
		void ResizeFont(int size);

		void SetColor(const Color& color) {m_color = color;}

		void Draw2D(const Vector2Df& pos, const char* text);

	protected:

	private:
		int m_currentSize;
		Color m_color;

		CLFile* m_fontFile;
		GLTexture* m_textureAtlas;
		texture_atlas_t* m_atlas;
		texture_font_t* m_font;
		vertex_buffer_t* m_buffer;

		static GLProgram* m_glProgram;

		PREVENT_COPYING(Font);
	};
}

#endif
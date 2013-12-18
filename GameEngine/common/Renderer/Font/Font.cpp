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

#include "Font.h"
#include "Utils/CLFileImporter.h"
#include "Renderer/Shader/GLProgram.h"

namespace CaptainLucha
{
	typedef struct {
		float x, y, z;    // position
		float s, t;       // texture
		float r, g, b, a; // color
	} vertex_t;

	Color ParseColor(const char* text, int index)
	{
		std::stringstream ss;
		for(int i = 0; i < 8; ++i)
			ss << text[index + 2 + i];
		
		unsigned int val;
		ss >> std::hex >> val;

		Color result;
		result.r = ((val >> 24) & 0xff) / 255.0f;
		result.g = ((val >> 16) & 0xff) / 255.0f;
		result.b = ((val >> 8) & 0xff) / 255.0f;
		result.a = ((val >> 0) & 0xff) / 255.0f;
		return result;
	}

	void add_text( vertex_buffer_t * buffer, texture_font_t * font,
		char* fileBuffer, int bufferSizeBytes,
		const char* text, vec2 * pen, int currentSize )
	{
		vec2 startPos = *pen;
		size_t length = strlen(text);
		Color vertColor = Color::White;

		for(size_t i=0; i < length; ++i)
		{
			if(text[i] == '\n')
			{
				pen->x = startPos.x;
				pen->y -= currentSize;
				continue;
			}

			if(text[i] == '<')
			{
				if(i + 1 < length && text[i + 1] == '#')
				{
					vertColor = ParseColor(text, i);
					i += 11;
				}
			}

			float r = vertColor.r;
			float g = vertColor.g;
			float b = vertColor.b;
			float a = vertColor.a;

			texture_glyph_t *glyph = texture_font_get_glyph( font, fileBuffer, bufferSizeBytes, text[i] );
			if( glyph != NULL )
			{
				float kerning = 0;
				if( i > 0)
				{
					kerning = texture_glyph_get_kerning( glyph, text[i-1] );
				}
				pen->x += kerning;
				int x0  = (int)( pen->x + glyph->offset_x );
				int y0  = (int)( pen->y + glyph->offset_y );
				int x1  = (int)( x0 + glyph->width );
				int y1  = (int)( y0 - glyph->height );
				float s0 = glyph->s0;
				float t0 = glyph->t0;
				float s1 = glyph->s1;
				float t1 = glyph->t1;
				GLuint indices[6] = {0,1,2, 0,2,3};
				vertex_t vertices[4] = { 
					{(float)x0, (float)y0, 1.0f, s0, t0, r, g, b, a},
					{(float)x0, (float)y1, 1.0f, s0, t1, r, g, b, a},
					{(float)x1, (float)y1, 1.0f, s1, t1, r, g, b, a},
					{(float)x1, (float)y0, 1.0f, s1, t0, r, g, b, a} };
				vertex_buffer_push_back( buffer, vertices, 4, indices, 6 );
				pen->x += glyph->advance_x;
			}
		}
	}

	Font::Font(const char* filepath, int size)
		: m_font(NULL),
		  m_atlas(NULL),
		  m_buffer(NULL),
		  m_textureAtlas(NULL),
		  m_color(Color::White),
		  m_currentSize(size)
	{
		REQUIRES(filepath)
		CLFileImporter fileImporter;
		m_fontFile = fileImporter.LoadFile(filepath);
		REQUIRES(m_fontFile->m_buffer && m_fontFile->m_bufLength > 0)

		m_buffer = vertex_buffer_new("vertex:3f,tex_coord:2f,color:4f");

		ResizeFont(size);

		if(!m_glProgram)
			m_glProgram = new GLProgram("Data/Shaders/FontShader.vert", "Data/Shaders/FontShader.frag");
	}

	Font::~Font()
	{
		texture_font_delete(m_font);
		texture_atlas_delete(m_atlas);
		vertex_buffer_delete(m_buffer);
		delete m_fontFile;
	}

	void Font::ResizeFont(int size)
	{
		if(m_font) texture_font_delete(m_font);
		if(m_atlas) texture_atlas_delete(m_atlas);

		delete m_textureAtlas;

		m_atlas = texture_atlas_new(512, 512, 1);
		m_font = texture_font_new(m_atlas, m_fontFile->m_buffer, m_fontFile->m_bufLength, (float)size);

		texture_font_load_glyphs(m_font, 
			L" !\"#$%&'()*+,-./0123456789:;<=>?"
			L"@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_"
			L"`abcdefghijklmnopqrstuvwxyz{|}~", 
			m_fontFile->m_buffer, m_fontFile->m_bufLength);

		m_textureAtlas = new GLTexture(m_atlas->id, 512, 512);
	}

	void Font::Draw2D(const Vector2Df& pos, const char* text)
	{
		add_text(m_buffer, m_font, m_fontFile->m_buffer, m_fontFile->m_bufLength, text, ((vec2*)&pos), m_currentSize);

		glEnable( GL_BLEND );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		m_glProgram->SetModelViewProjection();
		m_glProgram->SetUnifromTexture("diffuseMap", m_textureAtlas);
		m_glProgram->UseProgram();

		vertex_buffer_render(m_buffer, GL_TRIANGLES);
		vertex_buffer_clear(m_buffer);
	}

	GLProgram* Font::m_glProgram = NULL;
}
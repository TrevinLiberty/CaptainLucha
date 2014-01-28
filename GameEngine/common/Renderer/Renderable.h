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
 *	@file	Renderable.h
 *	@brief	
 *
/****************************************************************************/

#ifndef RENDERABLE_H_CL
#define RENDERABLE_H_CL

#include "Color.h"
#include "Shader/GLProgram.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
    class GLProgram;
    class Renderer;

	class Renderable
	{
	public:
        Renderable()
            : m_color(Color::White),
              m_isVisible(true),
              m_renderedOnAlphaPass(false),
              m_reflectable(true),
              m_castsShadows(true) {};
        ~Renderable() {};

        virtual void Draw(GLProgram& glProgram, Renderer& renderer) = 0;

        bool IsVisible() const {return m_isVisible;}
        void SetVisible(bool val) {m_isVisible = val;}

        bool IsRenderedOnAlphaPass() const {return m_renderedOnAlphaPass;}
        void SetRenderedOnAlphaPass(bool val) {m_renderedOnAlphaPass = val;}

        const Color& GetColor() const {return m_color;}
        void SetColor(const Color& val) {m_color = val;}

        const Vector3Df& GetPosition() const {return m_position;}
        void SetPosition(const Vector3Df& val) {m_position = val;}

        bool IsReflectable() const {return m_reflectable;}
        void SetReflectable(bool val) {m_reflectable = val;}

        bool CastsShadows() const {return m_castsShadows;}
        void SetCastsShadows(bool val) {m_castsShadows = val;}

	protected:

    private:
        bool m_isVisible;
        bool m_renderedOnAlphaPass;
        bool m_reflectable;
        bool m_castsShadows;

        Color m_color;
        Vector3Df m_position;
	};
}

#endif
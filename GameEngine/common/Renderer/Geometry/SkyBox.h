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
 *	@file	SkyBox.h
 *	@brief	
 *
/****************************************************************************/

#ifndef SKYBOX_H_CL
#define SKYBOX_H_CL

#include "Renderer/Geometry/Sphere.h"

namespace CaptainLucha
{
    class GLProgram;
    class Renderer;

	class SkyBox
	{
	public:
		SkyBox(
            const char* x1Path,
            const char* x2Path,
            const char* y1Path,
            const char* y2Path,
            const char* z1Path,
            const char* z2Path);
		~SkyBox();

        void SetCameraPosition(const Vector3Df& pos) {m_camPos = pos;}
        void Draw(Renderer& renderer);

	protected:

	private:
        GLProgram* m_program;
        GLTexture* m_cubeMap;

        Sphere m_sphere;
        Vector3Df m_camPos;

        unsigned int m_textureID;
	};
}

#endif
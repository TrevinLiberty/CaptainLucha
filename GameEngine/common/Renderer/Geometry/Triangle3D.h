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
 *	@file	Triangle3D.h
 *	@brief	
 *
/****************************************************************************/

#ifndef TRIANGLE3D_H_CL
#define TRIANGLE3D_H_CL

#include <Renderer/RendererUtils.h>
#include "Math/Vector3D.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class Triangle3D
	{
	public:
		Triangle3D(float scale = 1.0f);
		~Triangle3D();

		void Draw(GLProgram& glProgram);

	protected:
		void CreateGeometry(float scale);

	private:
		unsigned int m_vbo;
		unsigned int m_numverticesBuffer;
	};
}

#endif
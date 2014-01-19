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

#include "DebugGrid.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/VertexBufferObject.h"

namespace CaptainLucha
{
	DebugGrid::DebugGrid(int size, int padding)
	{
		const int HALF_SIZE = size / 2;

		std::vector<float> m_verts;

		for(int i = -HALF_SIZE; i <= HALF_SIZE; i += padding)
		{
			AddToVector(m_verts, -HALF_SIZE, 0.0f, (float)i);
			AddToVector(m_verts, HALF_SIZE, 0.0f, (float)i);

			AddToVector(m_verts, (float)i, 0.0f, -HALF_SIZE);
			AddToVector(m_verts, (float)i, 0.0f, HALF_SIZE);
		}

		m_vbo = new VertexBufferObject(m_verts, false, false, CL_LINES);
		SetObjectColor(Color::Blue);
	}

	DebugGrid::~DebugGrid()
	{
		delete m_vbo;
	}

	void DebugGrid::Draw(GLProgram& glProgram)
	{
		glProgram.SetUniform("emissive", 1.0);
		glProgram.SetUniform("color", GetColor());

		glProgram.SetUniform("hasDiffuseMap", false);
		glProgram.SetUniform("hasNormalMap", false);
		glProgram.SetUniform("hasSpecularMap", false);
		glProgram.SetUniform("hasEmissiveMap", false);

		g_MVPMatrix->Translate(0.0f, -1.0f, 0.0f);

		m_vbo->Draw(glProgram);
	}
}
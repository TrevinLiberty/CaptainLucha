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

#include "Cuboid.h"

#include "../RendererUtils.h"

namespace CaptainLucha
{
	const int NUM_INDICES = 36;

	Cuboid::Cuboid(const Vector3Df& extent)
	{
		CreateCuboid(extent.x * 2.0f, extent.y * 2.0f, extent.z * 2.0f);
	}

	Cuboid::Cuboid(float width, float height, float depth)
	{
		CreateCuboid(width, height, depth);
	}

	Cuboid::~Cuboid()
	{
		glDeleteBuffers(1, &m_vbo);
		glDeleteBuffers(1, &ibo_);
	}

	void Cuboid::Draw(GLProgram& glProgram)
	{
		glProgram.SetModelViewProjection();
		glProgram.UseProgram();

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);

		static const size_t STRIDE = sizeof(TangentSpaceVertex);

		const int vl = glProgram.GetAttributeLocation("vertex");
		const int nl = glProgram.GetAttributeLocation("normal");
		const int tl = glProgram.GetAttributeLocation("texCoord");

		const int tanl  = glProgram.GetAttributeLocation("tangent");
		const int btanl = glProgram.GetAttributeLocation("bitangent");

		EnableVertexAttrib(vl, 3, GL_FLOAT, GL_FALSE, STRIDE, 0);
		EnableVertexAttrib(nl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const GLvoid* >(offsetof(TangentSpaceVertex, nx_)));
		EnableVertexAttrib(tl, 2, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const GLvoid* >(offsetof(TangentSpaceVertex, u_)));

		EnableVertexAttrib(tanl,  3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const GLvoid* >(offsetof(TangentSpaceVertex, tstx_)));
		EnableVertexAttrib(btanl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const GLvoid* >(offsetof(TangentSpaceVertex, tsbx_)));

		glDrawElements(GL_TRIANGLES, NUM_INDICES, GL_UNSIGNED_INT, 0);

		DisableVertexAttrib(vl);
		DisableVertexAttrib(nl);
		DisableVertexAttrib(tl);
		DisableVertexAttrib(tanl);
		DisableVertexAttrib(btanl);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	void Cuboid::CreateCuboid(float width, float height, float depth)
	{
		TangentSpaceVertex vertices[] = {
			{-1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f},
			{-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f},
			{ 1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f},
			{ 1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f},

			{ 1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f},
			{ 1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f},
			{-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f},
			{-1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f},

			{ 1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f},
			{ 1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f},
			{ 1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f},
			{ 1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f},

			{-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f},
			{-1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f},
			{-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f},
			{-1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f},

			{1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f},
			{1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f},
			{-1.0f, 1.0f, -1.0f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f},
			{-1.0f, 1.0f,  1.0f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f},

			{ 1.0f, -1.0f, -1.0f, 0.0f, -1.0f,  0.0f,  0.0f,  1.0f},
			{ 1.0f, -1.0f,  1.0f, 0.0f, -1.0f,  0.0f,  0.0f,  0.0f},
			{-1.0f, -1.0f,  1.0f, 0.0f, -1.0f,  0.0f,  1.0f,  0.0f},
			{-1.0f, -1.0f, -1.0f, 0.0f, -1.0f,  0.0f,  1.0f,  1.0f},
		};

		static const unsigned int indices[] = {
			0,  1,  2,  
			0,  2,  3,
			4,  5,  6, 
			4,  6,  7,
			8,  9,  10, 
			8,  10, 11,
			12, 13, 14,
			12, 14, 15,
			16, 17, 18,
			16, 18, 19,
			20, 21, 22,
			20, 22, 23
		};

		for(int i = 0; i < 24; ++i)
		{
			vertices[i].x_ *= width;
			vertices[i].y_ *= height;
			vertices[i].z_ *= depth;
		}

		for(int i = 0; i < 12; ++i)
		{
			const int ROW = i * 3;

			SetTangentSpaceMatrix(vertices[indices[0 + ROW]], vertices[indices[1 + ROW]], vertices[indices[2 + ROW]], false);
			SetTangentSpaceMatrix(vertices[indices[1 + ROW]], vertices[indices[2 + ROW]], vertices[indices[0 + ROW]], false);
			SetTangentSpaceMatrix(vertices[indices[2 + ROW]], vertices[indices[0 + ROW]], vertices[indices[1 + ROW]], false);
		}

		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glGenBuffers(1, &ibo_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

}
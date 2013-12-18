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

#include "Triangle3D.h"

namespace CaptainLucha
{
	const int NUM_TRIANGLES = 7;

	Triangle3D::Triangle3D(float scale)
	{
		CreateGeometry(scale);
	}

	Triangle3D::~Triangle3D()
	{
		glDeleteBuffers(1, &m_vbo);
	}

	void Triangle3D::Draw(GLProgram& glProgram)
	{
		glProgram.SetModelViewProjection();
		int error = glGetError();
		if(error)
		{
			error += 10;
		}

		glProgram.UseProgram();

		error = glGetError();
		if(error)
		{
			error += 10;
		}

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

		const size_t STRIDE = sizeof(TangentSpaceVertex);

		const int vl = glProgram.GetAttributeLocation("vertex");
		const int nl = glProgram.GetAttributeLocation("normal");
		const int tl = glProgram.GetAttributeLocation("texCoord");
		const int tanl  = glProgram.GetAttributeLocation("tangent");
		const int btanl = glProgram.GetAttributeLocation("bitangent");

		EnableVertexAttrib(vl, 3, GL_FLOAT, GL_FALSE, STRIDE, 0);
		EnableVertexAttrib(nl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(TangentSpaceVertex, nx_)));
		EnableVertexAttrib(tl, 2, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(TangentSpaceVertex, u_)));
		EnableVertexAttrib(tanl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(TangentSpaceVertex, tstx_)));
		EnableVertexAttrib(btanl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(TangentSpaceVertex, tsbx_)));

		glDrawArrays(GL_TRIANGLES, 0, NUM_TRIANGLES * 3);

		DisableVertexAttrib(vl);
		DisableVertexAttrib(nl);
		DisableVertexAttrib(tl);
		DisableVertexAttrib(tanl);
		DisableVertexAttrib(btanl);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		error = glGetError();
		if(error)
		{
			error += 10;
		}
	}

	void Triangle3D::CreateGeometry(float scale)
	{
		const float HEIGHT = 1.5f;

		Vector3Df v1(1.5f, 0.0f, 0.0f);
		Vector3Df v2(-1.0f, 0.0f, 1.0f);
		Vector3Df v3(-1.0f, 0.0f, -1.0f);

		TangentSpaceVertex vertices[] = {
			{v3.x, HEIGHT,  v3.z,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f},
			{v2.x, HEIGHT,  v2.z,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f},
			{v1.x, HEIGHT,  v1.z,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f},

			{v2.x, 0.0f,   v2.z,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f},
			{v1.x, 0.0f,   v1.z,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f},
			{v1.x, HEIGHT, v1.z,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f},

			{v2.x, 0.0f,   v2.z,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f},
			{v1.x, HEIGHT, v1.z,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f},
			{v2.x, HEIGHT, v2.z,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f},

			{v3.x, 0.0f,   v3.z,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f},
			{v2.x, 0.0f,   v2.z,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f},
			{v2.x, HEIGHT, v2.z,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f},

			{v3.x, 0.0f,   v3.z,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f},
			{v2.x, HEIGHT, v2.z,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f},
			{v3.x, HEIGHT, v3.z,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f},

			{v1.x, 0.0f,   v1.z, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f},
			{v3.x, 0.0f,   v3.z, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f},
			{v3.x, HEIGHT, v3.z, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f},

			{v1.x, 0.0f,   v1.z, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f},
			{v3.x, HEIGHT, v3.z, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f},
			{v1.x, HEIGHT, v1.z, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f},
		};

		for(int i = 0; i < 21; ++i)
		{
			vertices[i].x_ *= scale;
			vertices[i].y_ *= scale;
			vertices[i].z_ *= scale;
		}

		for(int i = 0; i < 7; ++i)
		{
			const int vert = i * 3;
			SetTangentSpaceMatrix(vertices[vert], vertices[vert + 1], vertices[vert + 2], true);
			SetTangentSpaceMatrix(vertices[vert + 1], vertices[vert + 2], vertices[vert], true);
			SetTangentSpaceMatrix(vertices[vert + 2], vertices[vert + 1], vertices[vert], true);
		}

		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

}
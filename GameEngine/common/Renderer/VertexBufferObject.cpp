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

#include "VertexBufferObject.h"

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	VertexBufferObject::VertexBufferObject(const std::vector<float>& verts, bool usingColor, bool usingTexcoords, DrawType type)
		: isTangentSpaceVertices_(false),
		  isVectorFloat_(true),
		  usingColor_(usingColor),
		  usingTexCoords_(usingTexcoords),
		  drawType_(type)
	{
		REQUIRES(!verts.empty())

		int temp = 3;
		if(usingColor_)
			temp += 4;
		if(usingTexCoords_)
			temp += 2;
		size_ = verts.size() / temp;

		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * verts.size(), verts.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	VertexBufferObject::VertexBufferObject(const std::vector<Vertex>& verts)
		: isTangentSpaceVertices_(false),
		isVectorFloat_(false),
		usingColor_(false),
		drawType_(CL_TRIANGLES)
	{
		REQUIRES(!verts.empty())

		size_ = verts.size();

		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * size_, verts.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	VertexBufferObject::VertexBufferObject(const std::vector<TangentSpaceVertex>& verts)
		: isTangentSpaceVertices_(true),
		isVectorFloat_(false),
		drawType_(CL_TRIANGLES)
	{
		REQUIRES(!verts.empty())

		size_ = verts.size();

		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(TangentSpaceVertex) * size_, verts.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	VertexBufferObject::~VertexBufferObject()
	{
		glDeleteBuffers(1, &m_vbo);
	}

	void VertexBufferObject::Draw(GLProgram& glProgram) const
	{
		if(isVectorFloat_)
			DrawFloatVector(glProgram);
		else if(isTangentSpaceVertices_)
			DrawTangentSVertices(glProgram);
		else
			DrawVertices(glProgram);
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void VertexBufferObject::DrawFloatVector(GLProgram& glProgram) const
	{
		glProgram.SetModelViewProjection();
		glProgram.UseProgram();

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

		size_t STRIDE = 3;

		int vl = glProgram.GetAttributeLocation("vertex");
		int cl = glProgram.GetAttributeLocation("vertexColor");
		int tl = glProgram.GetAttributeLocation("texCoord");

		if (usingColor_)
			STRIDE += 4;
		if(usingTexCoords_)
			STRIDE += 2;
		STRIDE *= sizeof(float);

		int offset = 3;
		if (usingColor_)
			offset += 4;

		EnableVertexAttrib(vl, 3, GL_FLOAT, GL_FALSE, STRIDE, 0);
		EnableVertexAttrib(cl, 4, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offset * sizeof(float)));
		EnableVertexAttrib(tl, 2, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offset * sizeof(float)));

		if(drawType_ == CL_TRIANGLES)
			glDrawArrays(GL_TRIANGLES, 0, size_);
		else if(drawType_ == CL_LINES)
			glDrawArrays(GL_LINES, 0, size_);

		DisableVertexAttrib(vl);
		DisableVertexAttrib(cl);
		DisableVertexAttrib(tl);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void VertexBufferObject::DrawVertices(GLProgram& glProgram) const
	{
		glProgram.SetModelViewProjection();
		glProgram.UseProgram();

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

		const size_t STRIDE = sizeof(Vertex);

		const int vl = glProgram.GetAttributeLocation("vertex");
		const int nl = glProgram.GetAttributeLocation("normal");
		const int tl = glProgram.GetAttributeLocation("texCoord");

		EnableVertexAttrib(vl, 3, GL_FLOAT, GL_FALSE, STRIDE, 0);
		EnableVertexAttrib(nl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(Vertex, nx_)));
		EnableVertexAttrib(tl, 2, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(Vertex, u_)));

		if(drawType_ == CL_TRIANGLES)
			glDrawArrays(GL_TRIANGLES, 0, size_);
		else if(drawType_ == CL_LINES)
			glDrawArrays(GL_LINES, 0, size_);

		DisableVertexAttrib(vl);
		DisableVertexAttrib(nl);
		DisableVertexAttrib(tl);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void VertexBufferObject::DrawTangentSVertices(GLProgram& glProgram) const
	{
		glProgram.SetModelViewProjection();
		glProgram.UseProgram();

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

		if(drawType_ == CL_TRIANGLES)
			glDrawArrays(GL_TRIANGLES, 0, size_);
		else if(drawType_ == CL_LINES)
			glDrawArrays(GL_LINES, 0, size_);

		DisableVertexAttrib(vl);
		DisableVertexAttrib(nl);
		DisableVertexAttrib(tl);
		DisableVertexAttrib(tanl);
		DisableVertexAttrib(btanl);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
}
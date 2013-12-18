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

#include "IndexBufferObject.h"

namespace CaptainLucha
{
	IndexBufferObject::IndexBufferObject(const std::vector<Vertex>& verts,
		const std::vector<unsigned int>& indices)
		: VertexBufferObject(verts)
	{
		GenIBOBuffer(indices);
	}

	IndexBufferObject::IndexBufferObject(const std::vector<TangentSpaceVertex>& verts,
		const std::vector<unsigned int>& indices)
		: VertexBufferObject(verts)
	{
		GenIBOBuffer(indices);
	}

	IndexBufferObject::~IndexBufferObject()
	{

	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void IndexBufferObject::GenIBOBuffer(const std::vector<unsigned int>& indices)
	{
		REQUIRES(!indices.empty())

		numIndices_ = indices.size();

		glGenBuffers(1, &ibo_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * numIndices_, indices.data(), GL_STATIC_DRAW);
	}

	void IndexBufferObject::DrawVertices(GLProgram& glProgram) const
	{
		glProgram.SetModelViewProjection();
		glProgram.UseProgram();

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);

		static const size_t STRIDE = sizeof(Vertex);

		const int vl = glProgram.GetAttributeLocation("vertex");
		const int nl = glProgram.GetAttributeLocation("normal");
		const int tl = glProgram.GetAttributeLocation("texCoord");

		EnableVertexAttrib(vl, 3, GL_FLOAT, GL_FALSE, STRIDE, 0);
		EnableVertexAttrib(nl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(Vertex, nx_)));
		EnableVertexAttrib(tl, 2, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(Vertex, u_)));

		glDrawElements(GL_TRIANGLES, numIndices_, GL_UNSIGNED_INT, 0);

		DisableVertexAttrib(vl);
		DisableVertexAttrib(nl);
		DisableVertexAttrib(tl);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	void IndexBufferObject::DrawTangentSVertices(GLProgram& glProgram) const
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
		EnableVertexAttrib(nl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(TangentSpaceVertex, nx_)));
		EnableVertexAttrib(tl, 2, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(TangentSpaceVertex, u_)));
		EnableVertexAttrib(tanl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(TangentSpaceVertex, tstx_)));
		EnableVertexAttrib(btanl, 3, GL_FLOAT, GL_FALSE, STRIDE, reinterpret_cast<const void* >(offsetof(TangentSpaceVertex, tsbx_)));

		glDrawElements(GL_TRIANGLES, numIndices_, GL_UNSIGNED_INT, 0);

		DisableVertexAttrib(vl);
		DisableVertexAttrib(nl);
		DisableVertexAttrib(tl);
		DisableVertexAttrib(tanl);
		DisableVertexAttrib(btanl);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
}
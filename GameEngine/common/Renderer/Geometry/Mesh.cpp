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

#include "Mesh.h"

#include <Renderer/IndexBufferObject.h>

namespace CaptainLucha
{
	Mesh::Mesh(const std::vector<TangentSpaceVertex>& verts, const std::vector<unsigned int>& indices)
	{
		m_ibo = new IndexBufferObject(verts, indices);

		m_verts.resize(verts.size());
		m_indices.resize(indices.size());

		memcpy(m_verts.data(), verts.data(), sizeof(TangentSpaceVertex) * verts.size());
		memcpy(m_indices.data(), indices.data(), sizeof(unsigned int) * indices.size());
	}

	Mesh::~Mesh()
	{

	}

	void Mesh::Draw(GLProgram& glProgram)
	{
		m_ibo->Draw(glProgram);
	}

	AABoundingBox Mesh::GetAABB()
	{
		return AABoundingBox(m_verts);
	}
}
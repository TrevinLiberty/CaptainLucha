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
 *	@file	Mesh.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MESH_H_CL
#define MESH_H_CL

#include "Collision/CollisionListener.h"
#include "Collision/BoundingVolumes/AABoundingBox.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	struct TangentSpaceVertex;
	class IndexBufferObject;
	class GLProgram;

	class Mesh : public CollisionListener
	{
	public:
		Mesh(const std::vector<TangentSpaceVertex>& verts, const std::vector<unsigned int>& indices);
		~Mesh();

		void Draw(GLProgram& glProgram);

		AABoundingBox GetAABB();

	protected:

	private:
		IndexBufferObject* m_ibo;

		std::vector<TangentSpaceVertex> m_verts;
		std::vector<unsigned int> m_indices;
	};
}

#endif
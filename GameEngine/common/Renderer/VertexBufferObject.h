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
 *	@file	VertexBufferObject.h
 *	@brief	
 *
/****************************************************************************/

#ifndef VERTEX_BUFFER_OBJECT_H_CL
#define VERTEX_BUFFER_OBJECT_H_CL

#include "RendererUtils.h"

namespace CaptainLucha
{
	class VertexBufferObject
	{
	public:
		//Only draws GL_TRIANGLES
		VertexBufferObject(const std::vector<float>& verts, bool usingColor, bool usingTexcoords, DrawType type = CL_TRIANGLES);
		VertexBufferObject(const std::vector<Vertex>& verts);
		VertexBufferObject(const std::vector<TangentSpaceVertex>& verts);
		virtual ~VertexBufferObject();

		void Draw(GLProgram& glProgram) const;

	protected:
		virtual void DrawFloatVector(GLProgram& glProgram) const;
		virtual void DrawVertices(GLProgram& glProgram) const;
		virtual void DrawTangentSVertices(GLProgram& glProgram) const;

		unsigned int m_vbo;
		bool isTangentSpaceVertices_;
		bool isVectorFloat_;
		int size_;

		DrawType drawType_;

		bool usingColor_;
		bool usingTexCoords_;
	};
}

#endif
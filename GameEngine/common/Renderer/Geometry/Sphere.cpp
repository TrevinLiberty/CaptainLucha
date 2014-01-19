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

#include "Sphere.h"
#include "Utils/UtilDebug.h"

#include "Renderer/RendererUtils.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

#include <glew.h>

namespace CaptainLucha
{
	static const int NUM_FLOATS_PER_VERT = 6;

	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	Sphere::Sphere(float radius, int depth)
		: radius_(radius)
	{
		std::vector<TangentSpaceVertex> vertices;

		CreateSphereData(depth, vertices);

		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(TangentSpaceVertex) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		size = vertices.size();
	}
	
	Sphere::~Sphere()
	{

	}

	void Sphere::Draw(GLProgram& glProgram)
	{
		static const size_t STRIDE = sizeof(TangentSpaceVertex);

		glProgram.SetModelViewProjection();
		glProgram.UseProgram();

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

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

		glDrawArrays(GL_TRIANGLES, 0, size);

		DisableVertexAttrib(vl);
		DisableVertexAttrib(nl);
		DisableVertexAttrib(tl);
		DisableVertexAttrib(tanl);
		DisableVertexAttrib(btanl);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void Sphere::CreateSphereData(int depth, std::vector<TangentSpaceVertex>& vertices)
	{
		//The strange numbers X and Z are chosen so that the distance 
		//	from the origin to any of the vertices of the icosahedron is 1.0

		static const float X = .525731112119133606f;
		static const float Z = .850650808352039932f;

		static Vector3Df data[12][3] = {
			{Vector3Df(-X, 0.0, Z)}, {Vector3Df(X, 0.0, Z)}, {Vector3Df(-X, 0.0, -Z)}, {Vector3Df(X, 0.0, -Z)},    
			{Vector3Df(0.0, Z, X)}, {Vector3Df(0.0, Z, -X)}, {Vector3Df(0.0, -Z, X)}, {Vector3Df(0.0, -Z, -X)},    
			{Vector3Df(Z, X, 0.0)}, {Vector3Df(-Z, X, 0.0)}, {Vector3Df(Z, -X, 0.0)}, {Vector3Df(-Z, -X, 0.0)} 
		};

		static GLint tindices[20][3] = { 
			{0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},    
			{8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},    
			{7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6}, 
			{6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} 
		};

		for(int i = 0; i < 20; ++i)
		{
			SubDivide(
				data[tindices[i][0]][0],       
				data[tindices[i][1]][0],       
				data[tindices[i][2]][0], 
				depth, vertices); 
		}
	}
	
	void Sphere::SubDivide(const Vector3Df& v1, const Vector3Df& v2, const Vector3Df& v3, int depth, std::vector<TangentSpaceVertex>& vertices)
	{
		Vector3Df v12, v23, v31;
		
		if(depth == 0)
		{
			AddFace(v1, v2, v3, vertices);
		}
		else
		{
			for(int i = 0; i < 3; ++i)
			{
				v12[i] = v1[i] + v2[i];
				v23[i] = v2[i] + v3[i];
				v31[i] = v3[i] + v1[i];
			}

			v12.Normalize();
			v23.Normalize();
			v31.Normalize();

			//Four new faces were made
			//
			SubDivide(v1, v12, v31, depth - 1, vertices);
			SubDivide(v2, v23, v12, depth - 1, vertices);
			SubDivide(v3, v31, v23, depth - 1, vertices);
			SubDivide(v12, v23, v31, depth - 1, vertices);
		}
	}

	//TODO: Probably a better idea to have storage for normals and positions separate...
	//
	void Sphere::AddFace(const Vector3Df& v1, const Vector3Df& v2, const Vector3Df& v3, std::vector<TangentSpaceVertex>& vertices)
	{
		TangentSpaceVertex t1 = {v1.x * radius_, v1.y * radius_, v1.z * radius_, v1.x, v1.y, v1.z, 0.0f, 0.0f};
		TangentSpaceVertex t2 = {v2.x * radius_, v2.y * radius_, v2.z * radius_, v2.x, v2.y, v2.z, 0.0f, 0.0f};
		TangentSpaceVertex t3 = {v3.x * radius_, v3.y * radius_, v3.z * radius_, v3.x, v3.y, v3.z, 0.0f, 0.0f};

		vertices.push_back(t1); //Vertex
		vertices.push_back(t3);
		vertices.push_back(t2);
	}

	//////////////////////////////////////////////////////////////////////////
	//	Private
	//////////////////////////////////////////////////////////////////////////
}
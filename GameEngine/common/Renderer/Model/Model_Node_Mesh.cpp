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

#include "Model_Node_Mesh.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Shader/GLProgram.h"
#include "Renderer/Geometry/Mesh.h"
#include "Renderer/Model/Model.h"

namespace CaptainLucha
{
	Model_Node_Mesh::Model_Node_Mesh(const std::string& name, const Matrix4Df& transformation)
		: Model_Node(name, transformation)
	{

	}

	Model_Node_Mesh::~Model_Node_Mesh()
	{

	}

	void Model_Node_Mesh::Draw(GLProgram& glProgram, Model* owner)
	{
		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->MultMatrix(m_transformation);

		for(size_t i = 0; i < m_meshes.size(); ++i)
		{
			if(m_meshes[i].first)
			{
				owner->m_materials[m_meshes[i].second].ApplyMaterial(glProgram);
				m_meshes[i].first->Draw(glProgram);
			}
		}

		for(size_t i = 0; i < m_children.size(); ++i)
			m_children[i]->Draw(glProgram, owner);

		g_MVPMatrix->PopMatrix();
	}

	int Model_Node_Mesh::AddMesh(Mesh* newMesh, int materialIndex)
	{
		int returnIndex = -1;
		if(m_meshes.empty())
		{
			m_meshes.push_back(std::pair<Mesh*, int>(newMesh, materialIndex));
			returnIndex = 0;
		}
		else
		{
			for(size_t i = 0; i < m_meshes.size(); ++i)
			{
				if(!m_meshes[i].first)
				{
					m_meshes[i] = std::pair<Mesh*, int>(newMesh, materialIndex);
					returnIndex = i;
					break;
				}
			}
		}

		if(returnIndex == -1)
		{
			m_meshes.push_back(std::pair<Mesh*, int>(newMesh, materialIndex));
		}

		return returnIndex;
	}

	void Model_Node_Mesh::RemoveMesh(Mesh* mesh)
	{
		for(size_t i = 0; i < m_meshes.size(); ++i)
		{
			if(m_meshes[i].first == mesh)
				m_meshes[i].first = NULL;
		}
	}

	void Model_Node_Mesh::RemoveMesh(int index)
	{
		if(index >= 0 && index < (int)m_meshes.size())
		{
			m_meshes[index].first = NULL;
		}
	}

}
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

#include "Model_Node.h"

#include <Renderer/RendererUtils.h>

namespace CaptainLucha
{
	Model_Node::Model_Node(const std::string& name, const Matrix4Df& transformation)
		: m_name(name),
		  m_parent(NULL),
		  m_transformation(transformation)
	{}

	Model_Node::~Model_Node()
	{

	}

	void Model_Node::Draw(GLProgram& glProgram, Model* owner)
	{
		g_MVPMatrix->PushMatrix();
		//g_MVPMatrix->MultMatrix(m_transformation);

		for(size_t i = 0; i < m_children.size(); ++i)
			m_children[i]->Draw(glProgram, owner);

		g_MVPMatrix->PopMatrix();
	}

	void Model_Node::SetParent(Model_Node* parent)
	{
		m_parent = parent;
		if(m_parent)
			m_parent->m_children.push_back(this);
	}

	void Model_Node::GetAABB(AABoundingBox& currentBB)
	{
		for(size_t i = 0; i < m_children.size(); ++i)
			m_children[i]->GetAABB(currentBB);
	}
}
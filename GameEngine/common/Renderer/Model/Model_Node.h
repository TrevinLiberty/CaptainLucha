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
 *	@file	Model_Node.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MODEL_NODE_H_CL
#define MODEL_NODE_H_CL

#include "Math/Matrix4D.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class GLProgram;
	class Model;

	class Model_Node
	{
	public:
		Model_Node(const std::string& name, const Matrix4Df& transformation);
		~Model_Node();

		virtual void Draw(GLProgram& glProgram, Model* owner);

		Model_Node* GetParent() {return m_parent;}
		void SetParent(Model_Node* parent);

	protected:
		Model_Node* m_parent;
		std::vector<Model_Node*> m_children;

		Matrix4Df m_transformation;

		std::string m_name;

	private:

	};
}

#endif
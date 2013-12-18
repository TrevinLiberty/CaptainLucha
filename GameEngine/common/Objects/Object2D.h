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
 *	@file	Object2D.h
 *	@brief	
 *
/****************************************************************************/



#ifndef OBJECTTWOD_H_CL
#define OBJECTTWOD_H_CL

#include "Math/Vector3D.h"
#include "Renderer/Shader/GLProgram.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class GLProgram;

	class Object2D
	{
	public:
		Object2D();
		~Object2D();

		virtual void Update() {};
		virtual void Draw(GLProgram& glProgram) {UNUSED(glProgram)};
		virtual void DebugDraw() {};

		virtual const Vector2Df& GetPosition() {return m_position;}
		virtual void SetPosition(const Vector2Df& position) {m_position = position;}
		virtual void SetPosition(float x, float y) {m_position = Vector2Df(x, y);}

		int GetID() const {return m_id;}

		//don't use
		void SetID(int id) {m_id = id;}

	protected:
		Vector2Df m_position;

		void SetID() {m_id = m_validID++;}

	private:
		int m_id;

		static int m_validID;
	};
}

#endif
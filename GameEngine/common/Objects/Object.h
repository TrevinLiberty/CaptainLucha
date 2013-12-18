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
 *	@file	Object.h
 *	@brief	
 *
/****************************************************************************/



#ifndef OBJECT_H_CL
#define OBJECT_H_CL

#include "Math/Vector3D.h"
#include "Renderer/Shader/GLProgram.h"
#include "Renderer/Color.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class GLProgram;

	class Object
	{
	public:
		Object();
		~Object();

		virtual void Update(double DT) {UNUSED(DT)}
		virtual void Draw(GLProgram& glProgram) {UNUSED(glProgram)};
		virtual void DebugDraw() {};

		inline const Vector3D<Real>& GetPosition() const {return m_position;}
		inline void SetPosition(const Vector3D<Real>& position) {m_position = position;}
		inline void SetPosition(Real x, Real y, Real z) {m_position = Vector3D<Real>(x, y, z);}
		inline void AddToPosition(const Vector3D<Real>& add) {m_position += add;}

		const Color& GetColor() const {return m_color;}
		void SetObjectColor(const Color& val) {m_color = val;}

		bool IsVisible() const {return m_isVisible;}
		void SetVisible(bool val) {m_isVisible = val;}

		inline unsigned int GetID() const {return m_id;}

	protected:
		Vector3D<Real> m_position;
		Color m_color;

		void SetID() {m_id = m_validID++;}

	private:
		bool m_isVisible;

		unsigned int m_id;

		static unsigned int m_validID;
	};
}

#endif
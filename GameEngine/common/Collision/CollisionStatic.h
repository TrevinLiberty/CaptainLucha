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
 *	@file	CollisionStatic.h
 *	@brief	
 *
/****************************************************************************/

#ifndef COLLISIONSTATIC_H_CL
#define COLLISIONSTATIC_H_CL

#include "CollisionListener.h"
#include "Renderer/Geometry/Sphere.h"

namespace CaptainLucha
{
	class CollisionStatic : public CollisionListener
	{
	public:
		CollisionStatic();
		~CollisionStatic();

		/**
		 * @brief     Draws a debug cube or sphere representing this object
		 */
		virtual void Draw(GLProgram& glProgram);


		/**
		 * @brief     Initializes this static object as a cube.
		 */
		void InitCuboid(const Vector3D<Real>& position, const Quaternion& orientation, const Vector3Df& extent);

		/**
		 * @brief     Initializes this static object as a sphere.
		 */
		void InitSphere(const Vector3D<Real>& position, float radius);

	protected:

	private:
		Sphere m_drawSphere;
	};
}

#endif
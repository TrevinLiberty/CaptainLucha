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
 *	@file	StaticPlane.h
 *  @attention BROKEN.
 *  @todo	Broken. Only supports planes the have an up normal
 *	@brief	
 *
/****************************************************************************/

#ifndef STATICPLANE_H_CL
#define STATICPLANE_H_CL

#include "CollisionStatic.h"

namespace CaptainLucha
{
	class StaticPlane : public CollisionStatic
	{
	public:
		StaticPlane(const Vector3Df& pos, const Vector3Df& normal, const Vector3Df& extent);
		~StaticPlane();

		/**
		 * @brief     Draws a debug quad
		 */
		virtual void Draw(GLProgram& glProgram);

	protected:

	private:
		Vector3Df m_extent;
	};
}

#endif
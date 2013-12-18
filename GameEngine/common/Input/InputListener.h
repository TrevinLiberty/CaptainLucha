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
 *	@file	InputListener.h
 *	@brief	
 *
/****************************************************************************/

#ifndef INPUT_LISTENER_H_CL
#define INPUT_LISTENER_H_CL

#include "Utils/UtilMacros.h"

namespace CaptainLucha
{
	class InputListener
	{
	public:
		InputListener(int priorty = 0);
		~InputListener();

		virtual void KeyDown(int key) {UNUSED(key)};
		virtual void KeyUp(int key) {UNUSED(key)};
		virtual void MouseClick(int button, bool isDown) {UNUSED(button) UNUSED(isDown)};
		virtual void MouseMove(float x, float y) {UNUSED(x) UNUSED(y)}

		int GetPriority() const {return m_priority;}

	private:
		int m_priority;

		PREVENT_COPYING(InputListener);
	};
}

#endif
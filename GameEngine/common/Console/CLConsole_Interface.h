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
 *	@file	CLConsole_Interface.h
 *	@brief	Interface for a Console
 *  @see	CLConsole CLConsole_NULL
 *  @code 
		//Set to initially be the NULL interface. Allowing console functions to be called with no effect.
		CLConsole_Interface* myConsole = new CLConsole_NULL();

		//If console is desired
		delete myConsole;
		myConsole = new CLConsole();

		//Continue to use normally.
	@endcode
 *	@see	CLConsole CLConsole_NULL
 *  @todo	Change CONSOLE_WIDTH to percentage of screenwidth?
 *  @todo	Change CONSOLE_HEIGHT to be screenheight
/****************************************************************************/

#ifndef CONSOLE_INTERFACE_H_CL
#define CONSOLE_INTERFACE_H_CL

#include "Input/InputListener.h"
#include "Utils/UtilDebug.h"
#include "Utils/CommonIncludes.h"
#include "Threads/ThreadMutex.h"

namespace CaptainLucha
{
	class CLConsole_Interface
	{
	public:
		CLConsole_Interface() : m_isOpen(false) {};
		virtual ~CLConsole_Interface() {};

		virtual void Draw() = 0;

		virtual void AddHelpInfo(const char* command, const char* info) = 0;

		virtual void AddText(const char* text) = 0;
		virtual void AddErrorText(const char* text) = 0;
		virtual void AddSuccessText(const char* text) = 0;

		virtual void Open() {}
		virtual void Close() {}

		static const int CONSOLE_WIDTH = 750;
		static const int CONSOLE_HEIGHT = 920;

	protected:
		bool m_isOpen;

	private:
	};
}

#endif
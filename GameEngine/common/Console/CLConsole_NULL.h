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
 *	@file	CLConsole_NULL.h
 *	@brief	Used to disable the console functionality by creating this instead of CLConsole
 *
/****************************************************************************/

#ifndef CONSOLE_NULL_H_CL
#define CONSOLE_NULL_H_CL

namespace CaptainLucha
{
	class CLConsole_NULL : public CLConsole_Interface
	{
	public:
		CLConsole_NULL() {};
		virtual ~CLConsole_NULL() {};

		virtual void Draw() {};

		virtual void AddHelpInfo(const char* command, const char* info) {UNUSED(command) UNUSED(info)};

		virtual void AddText(const char* text) {UNUSED(text)};
		virtual void AddErrorText(const char* text) {UNUSED(text)};
		virtual void AddSuccessText(const char* text) {UNUSED(text)};

	protected:

	private:
	};
}

#endif
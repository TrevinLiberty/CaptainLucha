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
 *	@file	UtilDebug.h
 *	@brief	
 *
/****************************************************************************/

#ifndef UTIL_DEBUG_H_CL
#define UTIL_DEBUG_H_CL

#ifdef _WIN32
#define _WINSOCKAPI_
#include <Windows.h>
#include <WinBase.h>
#endif

#include <string>
#include <cassert>
#include <sstream>

namespace CaptainLucha
{
	inline void TraceMessage(const std::string& msg)
	{
		OutputDebugStringA(msg.c_str());
	}

	inline void MEMORY_LEAK_CHECK()
	{
#ifdef _WIN32
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif
	}

#define traceWithFile(expr)	\
	{	\
	std::ostringstream oss;	\
	oss << __FILE__ << "(" << __LINE__ << ") " << expr << " " << std::endl;	\
	CaptainLucha::TraceMessage(oss.str());	\
	}

#define trace(expr) \
	{	\
	std::ostringstream oss;	\
	oss << expr << std::endl;	\
	CaptainLucha::TraceMessage(oss.str());	\
	}

#define REQUIRES(eval)	\
	ASSERT_MSG(eval, "REQUIRES");

#define PROMISES(eval)	\
	ASSERT_MSG(eval, "PROMISE");

#define ASSERT_MSG(eval, msg)	\
	assert(eval && msg);
}

#endif
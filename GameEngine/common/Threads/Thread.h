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
 *	@file	Thread.h
 *	@brief	
 *
/****************************************************************************/

#ifndef CL_THREAD_H_CL
#define CL_THREAD_H_CL

#include "Utils/UtilMacros.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class ThreadImpl;
	class ThreadIDImpl;

	class ThreadID;

	typedef void(*ThreadFunction)(void*);

	ThreadID GetThisThreadID();
	void SleepThread(int milliseconds);
	unsigned int NumberOfHardwareThreads();

	class Thread
	{
	public:
		Thread(ThreadFunction func, void* args);
		~Thread();

		//See C++ 11 Join();
		void WaitForThreadToFinish();

		//See C++ 11 Joinable();
		bool CanWaitForThreadToFinish();

		//See C++ 11 Detach();
		void DetachFromThisThread();

		ThreadID GetThreadID() const;

		bool IsRunning() const { return m_isRunning; }
		void SetIsRunning(bool val) { m_isRunning = val; }

	private:
		Thread() {}

		ThreadImpl* m_impl;
		bool m_isRunning;

		PREVENT_COPYING(Thread)
	};

	class ThreadID
	{
	public:
		ThreadID();
		ThreadID(ThreadIDImpl* impl);
		~ThreadID();

		ThreadID & operator=(const ThreadID& aId);
		bool operator==(const ThreadID& aId2) const;
		bool operator!=(const ThreadID& aId2) const;

	private:
		ThreadIDImpl* m_impl;
	};
}

#endif
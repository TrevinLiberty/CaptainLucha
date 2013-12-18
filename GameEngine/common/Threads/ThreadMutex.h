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
 *	@file	ThreadMutex.h
 *	@brief	
 *
/****************************************************************************/

#ifndef THREAD_MUTEX_H_CL
#define THREAD_MUTEX_H_CL

#include "Utils/UtilMacros.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class ThreadMutexImpl1;
	class ThreadMutexImpl2;

	//////////////////////////////////////////////////////////////////////////
	//	ThreadMutex
	//////////////////////////////////////////////////////////////////////////
	class ThreadMutex
	{
	public:
		ThreadMutex();
		virtual ~ThreadMutex();

		virtual void Lock();
		virtual bool TryLock();
		virtual void Unlock();

	private:
		ThreadMutexImpl1* m_impl;

		PREVENT_COPYING(ThreadMutex)
	};

	//////////////////////////////////////////////////////////////////////////
	//	ThreadRecursiveMutex
	//////////////////////////////////////////////////////////////////////////
	class ThreadRecursiveMutex
	{
	public:
		ThreadRecursiveMutex();
		~ThreadRecursiveMutex();

		void Lock();
		bool TryLock();
		void Unlock();

	private:
		ThreadMutexImpl2* m_impl;

		PREVENT_COPYING(ThreadRecursiveMutex)
	};

	//////////////////////////////////////////////////////////////////////////
	//	ThreadLockGuard
	//////////////////////////////////////////////////////////////////////////
	template <typename MutexType>
	class ThreadLockGuard
	{
	public:
		ThreadLockGuard(MutexType& threadMutex)
			: m_mutex(threadMutex) 
		{
			m_mutex.Lock();
		}

		~ThreadLockGuard()
		{
			m_mutex.Unlock();
		}

	private:
		MutexType& m_mutex;

		PREVENT_COPYING(ThreadLockGuard)
	};
}

#endif
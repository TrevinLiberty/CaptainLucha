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
 *	@file	
 *	@brief	
 *
/****************************************************************************/

#include "ThreadMutex.h"

#include "tinythread.h"

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	ThreadMutexImpl1
	//////////////////////////////////////////////////////////////////////////
	class ThreadMutexImpl1
	{
	public:
		ThreadMutexImpl1() {};
		~ThreadMutexImpl1() {};

		inline void Lock()
		{
			m_mutex.lock();
		}

		inline bool TryLock()
		{
			return m_mutex.try_lock();
		}

		inline void Unlock()
		{
			m_mutex.unlock();
		}

	private:
		tthread::mutex m_mutex;
	};

	//////////////////////////////////////////////////////////////////////////
	//	ThreadMutexImpl2
	//////////////////////////////////////////////////////////////////////////
	class ThreadMutexImpl2
	{
	public:
		ThreadMutexImpl2() {};
		~ThreadMutexImpl2() {};

		inline void Lock()
		{
			m_mutex.lock();
		}

		inline bool TryLock()
		{
			return m_mutex.try_lock();
		}

		inline void Unlock()
		{
			m_mutex.unlock();
		}

	private:
		tthread::recursive_mutex m_mutex;
	};

	//////////////////////////////////////////////////////////////////////////
	//	ThreadMutex
	//////////////////////////////////////////////////////////////////////////
	ThreadMutex::ThreadMutex()
		: m_impl(new ThreadMutexImpl1())
	{
		
	}

	ThreadMutex::~ThreadMutex()
	{
		delete m_impl;
	}

	void ThreadMutex::Lock()
	{
		m_impl->Lock();
	}

	bool ThreadMutex::TryLock()
	{
		return m_impl->TryLock();
	}

	void ThreadMutex::Unlock()
	{
		m_impl->Unlock();
	}

	//////////////////////////////////////////////////////////////////////////
	//	ThreadMutex
	//////////////////////////////////////////////////////////////////////////
	ThreadRecursiveMutex::ThreadRecursiveMutex()
		: m_impl(new ThreadMutexImpl2())
	{

	}

	ThreadRecursiveMutex::~ThreadRecursiveMutex()
	{
		delete m_impl;
	}

	void ThreadRecursiveMutex::Lock()
	{
		m_impl->Lock();
	}

	bool ThreadRecursiveMutex::TryLock()
	{
		return m_impl->TryLock();
	}

	void ThreadRecursiveMutex::Unlock()
	{
		m_impl->Unlock();
	}
}
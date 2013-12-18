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

#include "Thread.h"

#include "tinythread.h"

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	ThreadIDImpl
	//////////////////////////////////////////////////////////////////////////	
	class ThreadIDImpl
	{
	public:
		/// Default constructor.
		/// The default constructed ThreadIDImpl is that of thread without a thread of
		/// execution.
		ThreadIDImpl() : m_id(0) {};

		ThreadIDImpl(unsigned long int aId) : m_id(aId) {};

		ThreadIDImpl(const ThreadIDImpl& aId) : m_id(aId.m_id) {};

		ThreadIDImpl(const tthread::thread::id& rhsID)
		{
			m_id = rhsID;
		}

		inline ThreadIDImpl & operator=(const ThreadIDImpl &aId)
		{
			m_id = aId.m_id;
			return *this;
		}

		inline friend bool operator==(const ThreadIDImpl &aId1, const ThreadIDImpl &aId2)
		{
			return (aId1.m_id == aId2.m_id);
		}

		inline friend bool operator!=(const ThreadIDImpl &aId1, const ThreadIDImpl &aId2)
		{
			return (aId1.m_id != aId2.m_id);
		}

	private:
		tthread::thread::id m_id;
	};

	//////////////////////////////////////////////////////////////////////////
	//	ThreadImpl
	//////////////////////////////////////////////////////////////////////////
	class ThreadImpl
	{
	public:
		ThreadImpl(ThreadFunction func, void* args)
			: m_thread(func, args)
		{

		}

		~ThreadImpl()
		{

		}

		//See C++ 11 Join();
		inline void WaitForThreadToFinish()
		{
			m_thread.join();
		}

		//See C++ 11 Joinable();
		inline bool CanWaitForThreadToFinish()
		{
			return m_thread.joinable();
		}

		//See C++ 11 Detach();
		inline void DetachFromThisThread()
		{
			m_thread.detach();
			m_thread.get_id();
		}

		inline ThreadIDImpl* GetThreadID() const
		{
			return new ThreadIDImpl(m_thread.get_id());
		}

	private:
		tthread::thread m_thread;
	};

	//////////////////////////////////////////////////////////////////////////
	//	CLThread
	//////////////////////////////////////////////////////////////////////////
	Thread::Thread( ThreadFunction func, void* args )
		: m_impl(new ThreadImpl(func, args)),
		  m_isRunning(true)
	{

	}

	Thread::~Thread()
	{
		delete m_impl;
	}

	void Thread::WaitForThreadToFinish()
	{
		m_impl->WaitForThreadToFinish();
	}

	bool Thread::CanWaitForThreadToFinish()
	{
		return m_impl->CanWaitForThreadToFinish();
	}

	void Thread::DetachFromThisThread()
	{
		m_impl->DetachFromThisThread();
	}

	ThreadID Thread::GetThreadID() const
	{
		return ThreadID(m_impl->GetThreadID());
	}

	//////////////////////////////////////////////////////////////////////////
	//	ThreadID
	//////////////////////////////////////////////////////////////////////////
	ThreadID::ThreadID()
		: m_impl(NULL)
	{

	}

	ThreadID::ThreadID(ThreadIDImpl* impl)
		: m_impl(impl)
	{

	}

	ThreadID::~ThreadID()
	{
		delete m_impl;
	}

	ThreadID& ThreadID::operator=(const ThreadID &aId)
	{
		*m_impl = *aId.m_impl;
		return *this;
	}

	bool ThreadID::operator==(const ThreadID &aId2) const
	{
		return *m_impl == *aId2.m_impl;
	}

	bool ThreadID::operator!=(const ThreadID &aId2) const
	{
		return *m_impl != *aId2.m_impl;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Functions
	//////////////////////////////////////////////////////////////////////////
	ThreadID GetThisThreadID()
	{
		return ThreadID(new ThreadIDImpl(tthread::this_thread::get_id()));
	}

	void SleepThread(int milliseconds)
	{
		tthread::this_thread::sleep_for(tthread::chrono::milliseconds(milliseconds));
	}

	unsigned int NumberOfHardwareThreads()
	{
		return tthread::thread::hardware_concurrency();
	}
}
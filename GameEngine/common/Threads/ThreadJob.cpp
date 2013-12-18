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

#include "ThreadJob.h"

#include "EventSystem/EventSystem.h"
#include "Utils/CLFileImporter.h"

#include <limits.h>
#include <fstream>

namespace CaptainLucha
{
	static int uniqueID = 0;
	static ThreadMutex uniqueIDMutex;

	ThreadJob::ThreadJob(const std::string& eventFireOnComplete, ThreadJobPriority priority)
		: m_eventFireOnComplete(eventFireOnComplete),
		  m_jobTag(""),
		  m_threadOwnerID(GetThisThreadID())
	{
		ThreadLockGuard<ThreadMutex> guard(uniqueIDMutex);
		m_uniqueJobID = uniqueID;
		++uniqueID;

		SetJobPriority(priority);
	}

	ThreadJob::ThreadJob(const std::string& eventFireOnComplete, const std::string& threadTag, ThreadJobPriority priority)
		: m_eventFireOnComplete(eventFireOnComplete),
		m_jobTag(threadTag),
		m_threadOwnerID(GetThisThreadID())
	{
		ThreadLockGuard<ThreadMutex> guard(uniqueIDMutex);
		m_uniqueJobID = uniqueID;
		++uniqueID;

		SetJobPriority(priority);
	}

	ThreadJob::~ThreadJob()
	{

	}

	void ThreadJob::SetJobPriority(int val)
	{
		m_priority = val;
	}

	void ThreadJob::SetJobPriority(ThreadJobPriority priority)
	{
		switch (priority)
		{
		case CL_PRIORITY_VERYLOW:
			m_priority = 0;
			break;
		case CL_PRIORITY_LOW:
			m_priority = 25;
			break;
		case CL_PRIORITY_NORMAL:
			m_priority = 50;
			break;
		case CL_PRIORITY_HIGH:
			m_priority = 75;
			break;
		case CL_PRIORITY_VERYHIGH:
			m_priority = 100;
			break;
		}
	}

	void ThreadJob::Completed(bool jobResult)
	{
		NamedProperties np = GetCompletedNamedProperties(jobResult);
		FireEvent(m_eventFireOnComplete, np);
	}

	NamedProperties ThreadJob::GetCompletedNamedProperties(bool jobResult) const
	{
		NamedProperties np;
		np.Set("JobID", m_uniqueJobID);
		np.Set("JobTag", m_jobTag);
		np.Set("Result", jobResult);
		return np;
	}
}
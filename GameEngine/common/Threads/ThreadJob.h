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
 *	@file	ThreadJob.h
 *	@brief	
 *
/****************************************************************************/

#ifndef THREAD_JOB_H_CL
#define THREAD_JOB_H_CL

#include "Utils/UtilMacros.h"
#include "Threads/Thread.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	enum ThreadJobPriority
	{
		CL_PRIORITY_VERYLOW,
		CL_PRIORITY_LOW,
		CL_PRIORITY_NORMAL,
		CL_PRIORITY_HIGH,
		CL_PRIORITY_VERYHIGH
	};
	
	class NamedProperties;
	struct CLFile;

	//////////////////////////////////////////////////////////////////////////
	//	ThreadJob
	//////////////////////////////////////////////////////////////////////////
	class ThreadJob
	{
	public:
		ThreadJob(const std::string& eventFireOnComplete, ThreadJobPriority priority = CL_PRIORITY_NORMAL);
		ThreadJob(const std::string& eventFireOnComplete, const std::string& jobTag, ThreadJobPriority priority = CL_PRIORITY_NORMAL);
		virtual ~ThreadJob();

		int GetUniqueJobID() const { return m_uniqueJobID; }
		const std::string& GetTag() const {return m_jobTag;}

		//Usually 0-100. VeryHigh priority->100
		void SetJobPriority(int val);
		void SetJobPriority(ThreadJobPriority priority);

		int GetPriority() const{return m_priority;}

		const ThreadID& ThreadOwnerID() const { return m_threadOwnerID; }

		virtual bool Execute() {return true;}
		virtual void Completed(bool jobResult);

	protected:
		NamedProperties GetCompletedNamedProperties(bool jobResult) const;

		ThreadJob() {}
		ThreadJob(const ThreadJob& job) {UNUSED(job)}

		int m_uniqueJobID;
		ThreadID m_threadOwnerID;
		
		std::string m_eventFireOnComplete;
		std::string m_jobTag;

		int m_priority;
	};
}

#endif
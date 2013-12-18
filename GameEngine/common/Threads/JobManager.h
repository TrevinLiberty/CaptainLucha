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
 *	@file	JobManager.h
 *	@brief	
 *
/****************************************************************************/

#ifndef JOB_MANAGER_H_CL
#define JOB_MANAGER_H_CL

#include "Utils/UtilDebug.h"

#include "Thread.h"
#include "ThreadMutex.h"
#include "ThreadJob.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class JobManager
	{
	public:
		static JobManager* GetInstance()
		{
			REQUIRES(m_manager)
				return m_manager;
		}

		static void CreateInstance()
		{
			if(!m_manager)
				m_manager = new JobManager();
		}

		static void DeleteInstance()
		{
			delete m_manager;
			m_manager = NULL;
		}

		void Update();

		void QueueNewJob(ThreadJob* newJob);
		void AddCompletedJob(ThreadJob* completedJob, bool jobResult);
		ThreadJob* TakeNewJob();

		void SetDesiredNumberOfThreads(int num) {REQUIRES(num >= 0) m_desiredNumThreads = num;}

		int GetDesiredNumberOfThreads();
		int GetActualNumberOfThreads();

		void ThreadStopped(int id);

		void WaitForThreadsAndDelete();

	private:
		static JobManager* m_manager;

		JobManager();
		~JobManager();

		std::list<ThreadJob*> m_queuedJobs;
		ThreadMutex m_queuedJobsMutex;

		std::vector<std::pair<ThreadJob*, bool> > m_completedJobs;
		ThreadMutex m_completedJobsMutex;

		std::vector<Thread*> m_threads;
		ThreadMutex m_threadVectorMutex;

		int m_desiredNumThreads;
		ThreadMutex m_desiredNumMutex;

		PREVENT_COPYING(JobManager)
	};
}

#endif
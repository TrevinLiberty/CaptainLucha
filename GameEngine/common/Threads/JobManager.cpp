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

#include "JobManager.h"
#include "Time/Time.h"

#include <algorithm>

namespace CaptainLucha
{
	static ThreadMutex* idMutex = new ThreadMutex();
	void JobThreadFunction(void* args)
	{
		UNUSED(args)
		int threadID = -1;

		idMutex->Lock();
		static int nextThreadID = 1;
		threadID = nextThreadID;
		++nextThreadID;
		idMutex->Unlock();

		ThreadJob* currentJob = NULL;
		for(;;)
		{
			currentJob = JobManager::GetInstance()->TakeNewJob();

			if(currentJob != NULL)
			{
				std::stringstream ss;
				ss << "THREAD " << threadID << " Starting on Job ";
				if(currentJob->GetTag().empty())
					ss << currentJob->GetUniqueJobID();
				else
					ss << "\"" << currentJob->GetTag() << "\"";
				ss << " Time: " << GetAbsoluteTimeSeconds();
				//Console::GetInstance()->AddText(ss.str(), Vector3Df(1.0f, 0.75f, 0.25f));

				bool jobResult = currentJob->Execute();
				JobManager::GetInstance()->AddCompletedJob(currentJob, jobResult);

				ss.str("");
				ss << "THREAD " << threadID << " Completed Job "; 
				if(currentJob->GetTag().empty())
					ss << currentJob->GetUniqueJobID();
				else
					ss << "\"" << currentJob->GetTag() << "\"";
				ss << " Time: " << GetAbsoluteTimeSeconds();
				//Console::GetInstance()->AddText(ss.str(), Vector3Df(1.0f, 0.75f, 0.25f));
			}

			if(threadID > JobManager::GetInstance()->GetDesiredNumberOfThreads())
			{
				break;
			}

			Sleep(10);
		}

		idMutex->Lock();
		--nextThreadID;
		JobManager::GetInstance()->ThreadStopped(threadID);
		idMutex->Unlock();
	}

	void JobManager::Update()
	{
		std::vector<std::pair<ThreadJob*, bool> > completedJobsCopy;
		const ThreadID thisThreadID = GetThisThreadID();

		m_completedJobsMutex.Lock();
		for(auto it = m_completedJobs.begin(); it != m_completedJobs.end();)
		{
			if(it->first->ThreadOwnerID() == thisThreadID)
			{
				completedJobsCopy.push_back((*it));
				it = m_completedJobs.erase(it);
			}
			else
			{
				++it;
			}
		}
		m_completedJobsMutex.Unlock();

		for(size_t i = 0; i < completedJobsCopy.size(); ++i)
		{
			completedJobsCopy[i].first->Completed(completedJobsCopy[i].second);
			delete completedJobsCopy[i].first;
		}

		m_threadVectorMutex.Lock();
		for(int i = m_threads.size(); i < m_desiredNumThreads; ++i)
		{
			m_threads.push_back(new Thread(&JobThreadFunction, NULL));
		}

		for(auto it = m_threads.begin(); it != m_threads.end();)
		{
			if(!(*it)->IsRunning())
			{
				delete (*it);
				it = m_threads.erase(it);
			}
			else
				++it;
		}
		m_threadVectorMutex.Unlock();

		if(m_desiredNumThreads == 0)
		{
			ThreadJob* newThread = TakeNewJob();
			if(newThread)
			{
				bool jobResult = newThread->Execute();
				JobManager::GetInstance()->AddCompletedJob(newThread, jobResult);
			}
		}
	}

	bool JobSortCompare(const ThreadJob* lhs, const ThreadJob* rhs)
	{
		return lhs->GetPriority() > rhs->GetPriority();
	}

	void JobManager::QueueNewJob( ThreadJob* newJob )
	{
		ThreadLockGuard<ThreadMutex> guard(m_queuedJobsMutex);
		m_queuedJobs.push_back(newJob);
		std::stable_sort(m_queuedJobs.begin(), m_queuedJobs.end(), JobSortCompare);
	}

	void JobManager::AddCompletedJob( ThreadJob* completedJob, bool jobResult )
	{
		ThreadLockGuard<ThreadMutex> guard(m_completedJobsMutex);
		m_completedJobs.push_back(std::pair<ThreadJob*, bool>(completedJob, jobResult));
	}

	ThreadJob* JobManager::TakeNewJob()
	{
		ThreadLockGuard<ThreadMutex> guard(m_queuedJobsMutex);
		if(m_queuedJobs.empty())
			return NULL;

		ThreadJob* newJob = m_queuedJobs.front();
		m_queuedJobs.pop_front();

		return newJob;
	}

	void JobManager::ThreadStopped(int id)
	{
		//ThreadLockGuard<ThreadMutex> guard(m_threadVectorMutex);
		(*(m_threads.begin() + id - 1))->SetIsRunning(false);
	}

	void JobManager::WaitForThreadsAndDelete()
	{
		m_desiredNumMutex.Lock();
		m_desiredNumThreads = -1;
		m_desiredNumMutex.Unlock();

		m_threadVectorMutex.Lock();
		for(auto it = m_threads.begin(); it != m_threads.end();)
		{
			(*it)->WaitForThreadToFinish();
			delete (*it);
			++it;
		}
		m_threadVectorMutex.Unlock();

		m_threadVectorMutex.Lock();
		m_threads.clear();
		m_threadVectorMutex.Unlock();
	}

	int JobManager::GetDesiredNumberOfThreads()
	{
		ThreadLockGuard<ThreadMutex> guard(m_desiredNumMutex);
		return m_desiredNumThreads;
	}

	int JobManager::GetActualNumberOfThreads()
	{
		ThreadLockGuard<ThreadMutex> guard(m_threadVectorMutex);
		int result = m_threads.size();
		return result;
	}

	JobManager* JobManager::m_manager;

	JobManager::JobManager()
		: m_desiredNumThreads(2)
	{
		const int MAX_NUMBER_HARDWARE_THREADS = NumberOfHardwareThreads();
		if(MAX_NUMBER_HARDWARE_THREADS > 0)
			m_desiredNumThreads = MAX_NUMBER_HARDWARE_THREADS;

#ifndef MULTITHREADED_APP
		m_desiredNumThreads = 0;
#endif
	}

	JobManager::~JobManager()
	{
		for(auto it = m_queuedJobs.begin(); it != m_queuedJobs.end(); ++it)
		{
			delete (*it);
		}

		for(auto it = m_completedJobs.begin(); it != m_completedJobs.end(); ++it)
		{
			delete (*it).first;
		}

		delete idMutex;
	}
}
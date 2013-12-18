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
 *	@file	CLLogger.h
 *	@brief	
 *
/****************************************************************************/

#ifndef CLLOGGER_H_CL
#define CLLOGGER_H_CL

#include "EventSystem/NamedProperties.h"
#include "Threads/Thread.h"
#include "Threads/ThreadMutex.h"

#include <ctime>
#include <fstream>

#include "Utils/CommonIncludes.h"

namespace
{
	struct LogStruct
	{
		time_t time;
		std::string msg;
		bool onFile;

		LogStruct(time_t t, const std::string& l)
			: time(t), msg(l), onFile(false) {}
	};
}

namespace CaptainLucha
{
	class CLCore;

	class CLLogger
	{
	public:
		static CLLogger* GetInstance()
		{
			REQUIRES(m_log)
			return m_log;
		}

		static void CreateInstance()
		{
			if(!m_log)
				m_log = new CLLogger();
		}

		static void DeleteInstance()
		{
			delete m_log;
			m_log = NULL;
		}

		void Log(const std::string& log);

		void OutputLogToFile();

		void StopLogging() 
			{ThreadLockGuard<ThreadMutex> guard(m_logMutex); m_stopLogging = true;}
		bool ShouldStopLogging() 
			{ThreadLockGuard<ThreadMutex> guard(m_logMutex); return m_stopLogging;}

	protected:

	private:
		static CLLogger* m_log;

		CLLogger();
		~CLLogger();

		Thread* m_outputThread;

		ThreadMutex m_logMutex;
		std::list<LogStruct> m_logs;
		std::fstream m_logfile;

		bool m_stopLogging;
	};
}

#endif
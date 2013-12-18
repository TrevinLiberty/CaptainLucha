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

#include "CLLogger.h"

namespace CaptainLucha
{
	void OutputLogThread(void*)
	{
		for(;;)
		{
			SleepThread(500);

			if(CLLogger::GetInstance()->ShouldStopLogging())
				return;

			CLLogger::GetInstance()->OutputLogToFile();
		}
	}

	void RemoveColorText(std::string& line)
	{
		std::string temp = line;
		line.clear();

		for(size_t i = 0; i < temp.size(); ++i)
		{
			if(temp[i] == '<')
			{
				if(i + 1 < temp.size())
				{
					if(temp[i + 1] == '#')
					{
						i += 11;
						continue;
					}
				}
			}

			line.push_back(temp[i]);
		}
	}

	void CLLogger::Log(const std::string& log)
	{
		UNUSED(log)
			std::string line = log;
		RemoveColorText(line);

		ThreadLockGuard<ThreadMutex> guard(m_logMutex);
		m_logs.push_back(LogStruct(time(0), line));

		if(m_logs.size() > 250)
			m_logs.pop_front();
	}

	void CLLogger::OutputLogToFile()
	{
		ThreadLockGuard<ThreadMutex> guard(m_logMutex);
		struct tm* ts;
		for(auto it = m_logs.begin(); it != m_logs.end(); ++it)
		{
			if(!it->onFile)
			{
				ts = localtime(&it->time);
				m_logfile << ts->tm_hour << ":" << ts->tm_min << ":" << ts->tm_sec
					<< "    " << it->msg << "\n";

				it->onFile = true;
			}
		}
	}

	CLLogger::CLLogger()
		: m_stopLogging(false)
	{
		m_outputThread = new Thread(&OutputLogThread, NULL);
		m_logfile.open("Log.txt", std::ios_base::out | std::ios_base::trunc);
	}

	CLLogger::~CLLogger()
	{
		m_logfile.close();
	}

	CLLogger* CLLogger::m_log = NULL;
}
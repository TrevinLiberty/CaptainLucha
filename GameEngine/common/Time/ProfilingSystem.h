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
 *	@file	ProfilingSystem.h
 *	@brief	
 *
/****************************************************************************/

#ifndef PROFILING_SYSTEM_H_CL
#define PROFILING_SYSTEM_H_CL

#include "Utils/UtilDebug.h"

#include <Threads/ThreadMutex.h>

#include <unordered_map>
#include <string>
#include <vector>

namespace CaptainLucha
{
	struct ProfilingReport
	{
		std::string m_id;
		int m_numReportsCurrentFrame;
		int m_numReportsAllTime;
		double m_currentFrameTimeSec;
		double m_averageFrameTimeSec;
		double m_maxFrameTimeSec;
		double m_maxFrameTimeAllTimeSec;

		ProfilingReport()
			: m_numReportsCurrentFrame(0),
			m_numReportsAllTime(0),
			m_currentFrameTimeSec(0.0),
			m_averageFrameTimeSec(-1.0),
			m_maxFrameTimeSec(0.0),
			m_maxFrameTimeAllTimeSec(0.0)
		{}
	};

	class Font;

	class ProfilingSystem
	{
	public:
		static const int FONT_HEIGHT = 14;

		static ProfilingSystem* GetInstance()
		{
			REQUIRES(m_pSystem)
			return m_pSystem;
		}

		static void CreateInstace()
		{
			delete m_pSystem;
			m_pSystem = new ProfilingSystem();
		}

		void InitNewFrame();
		void AddReport(const std::string& profileID, double seconds);
		void ResetSystem();

		void SetDesiredSecondsPerFrame(double seconds) {m_desiredSecPerFrame = seconds;}

		void DebugDraw();

		static void DeleteInstance()
		{
			delete m_pSystem;
			m_pSystem = NULL;
		}

	protected:
		ProfilingSystem();
		~ProfilingSystem();

	private:
		static ProfilingSystem* m_pSystem;

		double m_desiredSecPerFrame;

		double m_lastFrameTimeSeconds;

		std::unordered_map<unsigned int, ProfilingReport> m_reports;
		ThreadMutex m_reportsMutex;

		static bool ResetProfilerConsole(std::shared_ptr<std::vector<std::string>> args) 
		{args; m_pSystem->ResetSystem(); return true;}
	};
}

#endif
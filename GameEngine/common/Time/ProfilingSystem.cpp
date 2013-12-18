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

#include "ProfilingSystem.h"

#include "Time/Time.h"
#include "Utils/Utils.h"
#include <Renderer/RendererUtils.h>
#include <iomanip>

namespace CaptainLucha
{
	void ProfilingSystem::InitNewFrame()
	{
		m_lastFrameTimeSeconds = GetAbsoluteTimeSeconds();

		for(auto it = m_reports.begin(); it != m_reports.end(); ++it)
		{
			if((*it).second.m_averageFrameTimeSec < 0.0 && (*it).second.m_numReportsCurrentFrame > 0)
				(*it).second.m_averageFrameTimeSec = (*it).second.m_currentFrameTimeSec;
			else if((*it).second.m_numReportsCurrentFrame > 0)
				(*it).second.m_averageFrameTimeSec 
				= (*it).second.m_averageFrameTimeSec * 0.95 + (*it).second.m_currentFrameTimeSec * 0.05;

			(*it).second.m_numReportsCurrentFrame = 0;
			(*it).second.m_currentFrameTimeSec = 0.0;
			(*it).second.m_maxFrameTimeSec = 0.0;
		}
	}

	void ProfilingSystem::AddReport( const std::string& profileID, double seconds )
	{
		ThreadLockGuard<ThreadMutex> guard(m_reportsMutex);
		ProfilingReport& report = m_reports[StringHash(profileID)];

		if(report.m_id.empty())
			report.m_id = profileID;

		report.m_currentFrameTimeSec += seconds;
		++report.m_numReportsAllTime;
		++report.m_numReportsCurrentFrame;

		if(report.m_maxFrameTimeAllTimeSec < seconds)
			report.m_maxFrameTimeAllTimeSec = seconds;
	
		if(report.m_maxFrameTimeSec < seconds)
			report.m_maxFrameTimeSec = seconds;
	}

	void ProfilingSystem::ResetSystem()
	{
		for(auto it = m_reports.begin(); it != m_reports.end(); ++it)
		{
			(*it).second.m_numReportsCurrentFrame = 0;
			(*it).second.m_numReportsAllTime = 0;
			(*it).second.m_currentFrameTimeSec = 0.0;
			(*it).second.m_averageFrameTimeSec = -1.0;
			(*it).second.m_maxFrameTimeAllTimeSec = 0.0;
			(*it).second.m_maxFrameTimeSec = 0.0;
		}
	}

	void ProfilingSystem::DebugDraw()
	{
		std::stringstream ss;
		ss.precision(5);
		
		const double FRAME_TIME = GetAbsoluteTimeSeconds() - m_lastFrameTimeSeconds;

		int currentHeight = WINDOW_HEIGHT - 10;
		float currentX = WINDOW_WIDTH * 0.225f;

		for(auto it = m_reports.begin(); it != m_reports.end(); ++it)
		{
// 			if(currentHeight - FONT_HEIGHT < 0.0)
// 			{
// 				Draw2DDebugText(Vector2Df(currentX, WINDOW_HEIGHT - 10), ss.str().c_str());
// 				ss.str("");
// 
// 				currentHeight = WINDOW_HEIGHT - 50;
// 				currentX += 275;
// 			}

			ss << "<#C0C0C0FF>" << (*it).second.m_id << std::setw(10) << std::setprecision(4)
				<< "\n<#FFFFFFFF>    Calls F/A: " << std::fixed << (*it).second.m_numReportsCurrentFrame  << " / " << (*it).second.m_numReportsAllTime
			   << " | " << (*it).second.m_currentFrameTimeSec / FRAME_TIME << "%"
			   << " | cTime: " << (*it).second.m_currentFrameTimeSec * 1000.0	<< "ms"
               << " | aTime: " << (*it).second.m_averageFrameTimeSec * 1000.0	<< "ms"
			   << " | MaxTCF : " << (*it).second.m_maxFrameTimeSec * 1000.0		<< "ms"
			   << " | MaxTAll: " << (*it).second.m_maxFrameTimeAllTimeSec * 1000.0 << "ms\n\n";

			currentHeight -= FONT_HEIGHT * 8;
		}

		if(!ss.str().empty())
			Draw2DDebugText(Vector2Df(currentX, WINDOW_HEIGHT - 10), ss.str().c_str());
		ss.str("");
	}

	ProfilingSystem::ProfilingSystem()
	{

	}

	ProfilingSystem::~ProfilingSystem()
	{

	}

	ProfilingSystem* ProfilingSystem::m_pSystem = NULL;

}
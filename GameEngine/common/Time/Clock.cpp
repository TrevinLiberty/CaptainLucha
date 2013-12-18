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

#include "Clock.h"

#include "EventSystem/EventSystem.h"

#include <limits>

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	Clock::Clock()
		: m_currentTime(0.0),
		  m_deltaTime(0.0),
		  m_timeScale(1.0),
		  m_isPaused(false),
		  m_parent(NULL)
	{
		RemoveMaxDT();
	}

	Clock::~Clock()
	{
		if(m_parent != NULL)
		{
			m_parent->RemoveChildClock(this);

			for(auto it = m_children.begin(); it != m_children.end(); ++it)
			{
				m_parent->m_children.push_back(*it);
			}

			for(auto it = m_timers.begin(); it != m_timers.end(); ++it)
			{
				(*it)->SetParentClock(m_parent);
				m_parent->m_timers.push_back(*it);
			}
		}
	}

	void Clock::RemoveMaxDT()
	{
		m_maxDT = std::numeric_limits<double>::infinity();
	}

	Timer* Clock::CreateNewTimer()
	{
		m_timers.push_back(new Timer(this));
		return m_timers.back();
	}

	void Clock::RemoveTimer(Timer* timer)
	{
		for(auto it = m_timers.begin(); it != m_timers.end(); ++it)
		{
			if((*it) == timer)
			{
				m_timers.erase(it);
				break;
			}
		}
	}

	void Clock::AddAlarmDelay(double secondsDelay, const std::string& event)
	{
		NamedProperties np;
		AddAlarmAbsolute(m_currentTime + secondsDelay, event, np);
	}

	void Clock::AddAlarmDelay(double secondsDelay, const std::string& event, NamedProperties& np)
	{
		AddAlarmAbsolute(m_currentTime + secondsDelay, event, np);
	}

	void Clock::AddAlarmAbsolute(double timeSeconds, const std::string& event)
	{
		NamedProperties np;
		AddAlarmAbsolute(timeSeconds, event, np);
	}

	bool CompareAlarms(const AlarmEvent& lhs, const AlarmEvent& rhs)
	{
		return lhs.m_fireTime < rhs.m_fireTime;
	}

	void Clock::AddAlarmAbsolute(double timeSeconds, const std::string& event, NamedProperties& np)
	{
		if(timeSeconds <= m_currentTime)
		{
			FireEvent(event, np);
			return;
		}

		m_alarms.push_back(AlarmEvent());
		auto& newAlarm = m_alarms.back();
		newAlarm.m_event = event;
		newAlarm.m_np = np;
		newAlarm.m_fireTime = timeSeconds;
		m_alarms.sort(CompareAlarms);
	}

	void Clock::Update( double DT )
	{
		double tempScale = m_timeScale;

		m_deltaTime = (DT > m_maxDT) ? m_maxDT : DT;

		if(m_isPaused)
			tempScale = 0.0;

		m_deltaTime *= tempScale;
		m_currentTime += m_deltaTime;

		for(auto it = m_children.begin(); it != m_children.end(); ++it)
			(*it)->Update(m_deltaTime);

		for(auto it = m_timers.begin(); it != m_timers.end(); ++it)
			(*it)->Update(m_deltaTime);

		while(!m_alarms.empty())
		{
			if(m_alarms.front().m_fireTime <= m_currentTime)
			{
				FireEvent(m_alarms.front().m_event, m_alarms.front().m_np);
				m_alarms.pop_front();
			}
			else
				break;
		}
	}

	Clock* Clock::CreateNewClock()
	{
		Clock* newClock = new Clock();
		newClock->m_parent = this;
		newClock->m_currentTime = m_currentTime;

		m_children.push_back(newClock);
		return newClock;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void Clock::RemoveChildClock(Clock* child)
	{
		for(auto it = m_children.begin(); it != m_children.end(); ++it)
		{
			if((*it) == child)
			{
				m_children.erase(it);
				return;
			}
		}
	}
}
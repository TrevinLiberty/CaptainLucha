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
 *	@file	Clock.h
 *	@brief	
 *
/****************************************************************************/

#ifndef CLOCK_H_CL
#define CLOCK_H_CL

#include "EventSystem/NamedProperties.h"
#include "Timer.h"

#include <Utils/CommonIncludes.h>

namespace
{
	struct AlarmEvent
	{
		std::string m_event;
		CaptainLucha::NamedProperties m_np;
		double m_fireTime;
	};
}

namespace CaptainLucha
{
	class Clock
	{
	public:
		Clock();
		~Clock();

		double GetCurrentTimeSeconds() const { return m_currentTime; }
		void SetCurrentTime(double seconds) { m_currentTime = seconds; }
		void ResetCurrentTime() {m_currentTime = 0.0;}

		double DeltaTime() const { return m_deltaTime; }

		double GetTimeScale() const { return m_timeScale; }
		void SetTimeScale(double val) { m_timeScale = val; }

		bool IsPaused() const { return m_isPaused; }
		void Pause() {m_isPaused = true;}
		void UnPause() {m_isPaused = false;}

		void SetMaxDT(double val) { m_maxDT = val; }
		void RemoveMaxDT();

		Timer* CreateNewTimer();
		void RemoveTimer(Timer* timer);

		void AddAlarmDelay(double secondsDelay, const std::string& event);
		void AddAlarmDelay(double secondsDelay, const std::string& event, NamedProperties& np);

		void AddAlarmAbsolute(double timeSeconds, const std::string& event);
		void AddAlarmAbsolute(double timeSeconds, const std::string& event, NamedProperties& np);

		void Update(double DT);

		Clock* CreateNewClock();

	protected:
		void RemoveChildClock(Clock* child);

	private:
		double m_currentTime;
		double m_deltaTime;

		double m_timeScale;
		double m_maxDT;

		bool m_isPaused;

		std::list<AlarmEvent> m_alarms;

		Clock* m_parent;
		std::list<Clock*> m_children;
		std::list<Timer*> m_timers;
	};
}

#endif
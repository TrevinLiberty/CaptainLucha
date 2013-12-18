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
 *	@file	Timer.h
 *	@brief	
 *
/****************************************************************************/

#ifndef TIMER_H_CL
#define TIMER_H_CL

#include "Utils/UtilMacros.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class Clock;

	class Timer
	{
	public:
		Timer(Clock* parentClock);
		~Timer();

		void Update(double dt);

		void SetTimer(double seconds);

		void StartTimer();
		bool IsTimeUp();

		bool IsOn() const {return m_isOn;}
		void Stop() {m_isOn = false;}

		float GetElapsedTime() const;

		void SetParentClock(Clock* parent) {m_parentClock = parent;}

		Timer& operator= (const Timer& rhs)
		{
			m_elapseTime = rhs.m_elapseTime;
			m_startTime = rhs.m_startTime;
			m_endTime = rhs.m_endTime;
			m_isOn = rhs.m_isOn;

			return *this;
		}

	protected:
		double m_currentTime;
		double m_elapseTime;

		double m_startTime;
		double m_endTime;

		bool m_isOn;

		Clock* m_parentClock;
	};
}

#endif
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

#include "Timer.h"
#include "Time.h"

#include "Clock.h"
#include "Utils/UtilDebug.h"

namespace CaptainLucha
{
	Timer::Timer(Clock* parentClock)
		: m_elapseTime(0),
		m_startTime(0),
		m_endTime(0),
		m_currentTime(0),
		m_isOn(false),
		m_parentClock(parentClock)
	{}

	Timer::~Timer()
	{
		m_parentClock->RemoveTimer(this);
	}

	void Timer::Update(double dt)
	{
		if(IsOn())
		{
			m_currentTime += dt;
		}
	}

	void Timer::SetTimer(double seconds)
	{
		m_elapseTime = seconds;
	}

	void Timer::StartTimer()
	{
		m_startTime = m_currentTime;
		m_endTime = m_startTime + m_elapseTime;
		m_isOn = true;
	}

	float Timer::GetElapsedTime() const 
	{
		return (float)(m_currentTime - m_startTime);
	}

	bool Timer::IsTimeUp()
	{
		if(m_currentTime >= m_endTime && m_isOn)
			return true;

		return false;
	}
}
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

#include "Time.h"

#include <Threads/ThreadMutex.h>

#define _WINSOCKAPI_
#include <Windows.h>

namespace CaptainLucha
{
	double g_invFreq = 0.0;
	ThreadMutex g_invFreqMutex;

	void InitTime()
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);

		g_invFreq = 1.0 / static_cast<double>(freq.QuadPart);
	}

	double GetAbsoluteTimeSeconds()
	{
		LARGE_INTEGER clocks;
		QueryPerformanceCounter(&clocks);

		ThreadLockGuard<ThreadMutex> guard(g_invFreqMutex);
		return (clocks.QuadPart * g_invFreq);
	}

	double GetClockFrequencyDouble()
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);

		return static_cast<double>(freq.QuadPart);
	}

	double ClocksToSeconds(SystemClocks clocks)
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);

		return clocks * (1.0 / static_cast<double>(freq.QuadPart));
	}

	SystemClocks GetAbsoluteTimeClocks()
	{
		LARGE_INTEGER clocks;
		QueryPerformanceCounter(&clocks);

		return clocks.QuadPart;
	}

	SystemClocks GetClockFrequency()
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);

		return freq.QuadPart;
	}

	SystemClocks SecondsToClocks(double seconds)
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);

		return static_cast<SystemClocks>(freq.QuadPart * seconds);
	}
}
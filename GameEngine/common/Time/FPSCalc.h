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
 *	@file	FPSCalc.h
 *	@brief	
 *
/****************************************************************************/

#ifndef FPSCALC_H_CL
#define FPSCALC_H_CL

namespace CaptainLucha
{
	class FPSCalc
	{
	public:
		FPSCalc();
		~FPSCalc();

		void Update();

		double GetFPS() const {return m_fps;}
		double GetRunningAverage() const {return m_runningAverage;}
		double GetMinFPS() const {return m_minFPS;}
		double GetMaxFPS() const {return m_maxFPS;}

	private:
		double m_fps;
		double m_runningAverage;
		double m_minFPS;
		double m_maxFPS;

		double m_previousTime;
		int m_frameCount;
	};
}

#endif
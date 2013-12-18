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
 *	@file	Utils.h
 *	@brief	
 *
/****************************************************************************/

#ifndef UTILS_H_CL
#define UTILS_H_CL

#include "UtilDebug.h"

#ifdef _WIN32
#define _WINSOCKAPI_
#include <Windows.h>
#include <WinBase.h>
#endif

#include "Math/Math.h"

#include <sstream>
#include <vector>
#include <fstream>
#include <iterator>
#include <ostream>
#include <iostream>

#include <glew.h>


namespace CaptainLucha
{
	enum CL_BaseTypes
	{
		CL_INT,
		CL_BOOL,
		CL_CHAR,
		CL_FLOAT,
		CL_DOUBLE,
		CL_UINT,
		CL_COLOR,
		CL_STRING,
		CL_VECTOR,
		CL_QUAT
	};

	std::string GetCurrentDirectoryPath_w();
	int GetLineNumberFromByteOffset(const std::string& fileName, int byteOffset);
	void OutputErrorMessage(const std::string& msg);
	void CreateDumbyFiles(const std::string& name, int numFiles, int numKBytes);
	std::string GetClipboardData_w();
	std::string GetComputerName_w();

	inline float DegreesToRadians(float degrees)
	{
		return degrees / 180.0f * Math<float>::PI;
	}

	inline float RadiansToDegrees(float radians)
	{
		return radians / MathF::PI * 180;
	}

	inline int RandInRangeInt(int min, int max)
	{
		const int r = rand() % (max - min);
		return r + min;
	}

	inline float RandInRange(float min, float max)
	{
		const float r = rand() / (float)RAND_MAX;
		return min + (max - min) * r;
	}

	inline int RandNeg() 
	{
		return rand()%2 == 0 ? -1 : 1;
	}

	template<typename T, typename Alpha>
	inline T Lerp(T min, T max, Alpha val)
	{
		return static_cast<T>(min + (max - min) * val);
	}

	inline float Map(float number, float inMin, float inMax, float outMin, float outMax)
	{
		return (number - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
	}

	template< typename real >
	real clamp( real value, real min, real max )
	{
		return value < min ? min : value > max ? max : value;
	}

	inline std::vector<std::string> DelimitString(const std::string& line, char delimiter)
	{
		std::stringstream ss(line);
		std::string value;
		std::vector<std::string> values;

		while(getline(ss, value, delimiter))
			values.push_back(value);

		return values;
	}

	inline unsigned int ByteHash(const char* s, int size)
	{
		unsigned int result = 0;

		for(int i = 0; i < size; ++i)
		{
			result &= 0x07ffffff;
			result *= 31;
			result += s[i];
		}

		return result;
	}

	inline unsigned int StringHash(const std::string& s)
	{
		return ByteHash(s.c_str(), s.size());
	}

	template<typename T>
	inline T FastInvSQRT(T val)
	{
		T halfVal = val * 0.5f;
		int i = *(int*)&val;
		i = 0x5f3759df - (i >> 1);
		val = *(T*)&i;
		val *= 1.5f - halfVal*val*val;
		return val;
	}

	inline bool IsPower2(int val)
	{
		return (val > 0) && ((val & (val - 1)) == 0);
	}

	void TokenizeString(const std::string& input, bool keepQuotedText, std::vector<std::string>& outTokens);
}

#endif
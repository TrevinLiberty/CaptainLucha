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

#include "WindowsUtils.h"

#define _WINSOCKAPI_
#include <Windows.h>

namespace CaptainLucha
{
	LPWIN32_FIND_DATA g_findFileData;
	HANDLE g_find;

	//Returns the first file with name, fileName
	std::string GetFirstFile(const char* fileName)
	{
		g_find = FindFirstFile(fileName, g_findFileData);
		if(g_find != INVALID_HANDLE_VALUE)
		{
			return g_findFileData->cFileName;
		}

		return "";
	}

	//Returns the name of the next file after the return from GetFirstFile
	//
	std::string GetNextFile()
	{
		if(g_find != INVALID_HANDLE_VALUE && FindNextFile(g_find, g_findFileData))
		{
			return g_findFileData->cFileName;
		}

		return "";
	}

	void CloseFileFind()
	{
		FindClose(g_find);
	}
}
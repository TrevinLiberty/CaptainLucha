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

#include "Utils.h"

namespace CaptainLucha
{
	std::string GetCurrentDirectoryPath_w()
	{
#ifdef _WIN32
		char result[MAX_PATH];
		GetCurrentDirectoryA(MAX_PATH, result);
		return result;
#endif
	}

	int GetLineNumberFromByteOffset(const std::string& fileName, int byteOffset)
	{
		std::ifstream file(fileName.c_str());

		int numLines = 0;

		if(file.is_open())
		{
			std::string fileString((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

			for(int i = 0; i < byteOffset; ++i)
			{
				if(fileString[i] == '\n')
					numLines++;
			}
		}

		return numLines;
	}


	void OutputErrorMessage(const std::string& msg)
	{
		//GLFW glutHideWindow();
		std::cout << msg << std::endl;
		MessageBox(NULL, msg.c_str(),"XML ERROR", MB_OK | MB_ICONERROR);
		//GLFW glutShowWindow();

		DebugBreak();
	}

	void CreateDumbyFiles(const std::string& name, int numFiles, int numKBytes)
	{
		if(numFiles <= 0 || numKBytes <= 0)
			return;

		std::stringstream ss;
		ss << name << "_" << numKBytes << "KB_File_";

		char* temp = new char[numKBytes * 1024];
		char buff[33];

		for(int i = 0; i < numFiles; ++i)
		{
			_itoa(i + 1, buff, 10);
			std::ofstream file(ss.str() + buff + ".dat", std::ios::out | std::ios::trunc);
			file.write(temp, numKBytes * 1024);
			file.close();
		}
	}

	std::string GetClipboardData_w()
	{
		std::string result;
		if(OpenClipboard(NULL)) 
		{
			HANDLE handle = GetClipboardData(CF_TEXT);
			if(handle)
			{
				result = (char*)handle;
			}
		}

		return result;
	}

	std::string GetComputerName_w()
	{
		char computerName[MAX_COMPUTERNAME_LENGTH + 1];
		DWORD t = MAX_COMPUTERNAME_LENGTH + 1;
		GetComputerNameA(computerName, &t);
		return computerName;
	}

	void TokenizeString(const std::string& input, bool keepQuotedText, std::vector<std::string>& outTokens)
	{
		outTokens.push_back(std::string());

		std::stringstream ss;
		ss << std::noskipws;
		ss << input;

		char letter;

		while(ss >> letter)
		{
			if(letter == ' ')
			{
				if(!outTokens.back().empty())
					outTokens.push_back(std::string());
			}
			else if(letter == '"' && keepQuotedText)
			{
				while(ss >> letter)
				{
					if(letter == '"')
						break;
					outTokens.back().push_back(letter);
				}
			}
			else
				outTokens.back().push_back(letter);
		}
	}
}
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

#include "Commandlets.h"
#include "EventSystem/EventSystem.h"
#include "Utils/Utils.h"

#include <sstream>
#include <algorithm>

#include <iostream>

namespace CaptainLucha
{
	Commandlets::Commandlets()
	{

	}

	Commandlets::~Commandlets()
	{

	}

	bool Commandlets::ParseCommands( int argc, char** argv )
	{
		if(argc < 2)
			return false;

		std::stringstream ss;
		ss << argv[1];

		std::string word;
		ss >> word;

		for(int i = 1; i < argc; ++i)
		{
			if(argv[i][0] == '-')
			{
				m_commands.push_back(std::pair<std::string, std::vector<std::string> >());
				m_commands.back().first = (word.c_str() + 1);

				++i;
				while(i < argc && argv[i][0] != '-')
				{
					m_commands.back().second.push_back(argv[i]);
					++i;
				}
			}
		}

		return PreGameUseCommands();
	}

	void Commandlets::FireCommandlets()
	{
		for(size_t i = 0; i < m_commands.size(); ++i)
		{
			const std::vector<std::string>* args = &m_commands[i].second;
			NamedProperties np;
			np.Set("args", args);
			FireEvent(m_commands[i].first, np);
		}
	}

	const std::vector<std::string>* Commandlets::GetCommandArguements(const std::string& command) const
	{
		for(size_t i = 0; i < m_commands.size(); ++i)
		{
			if(_strcmpi(m_commands[i].first.c_str(), command.c_str()) == 0)
			{
				return &m_commands[i].second;
			}
		}

		return NULL;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	bool Commandlets::PreGameUseCommands()
	{
		bool result = false;
		for(size_t i = 0; i < m_commands.size(); ++i)
		{
			std::cout << m_commands[i].first << std::endl;
			if(_strcmpi(m_commands[i].first.c_str(), "generateFiles") == 0)
			{
				if(m_commands[i].second.size() == 2)
				{
					int numFiles = atoi(m_commands[i].second[0].c_str());
					int numBytes = atoi(m_commands[i].second[1].c_str());

					CreateDumbyFiles("Data/generated", numFiles, numBytes);
					result = true;
				}
			}
		}

		return result;
	}
}
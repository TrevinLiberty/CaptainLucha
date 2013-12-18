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
 *	@file	Commandlets.h
 *	@brief	
 *
/****************************************************************************/

#ifndef COMMANDLETS_H_CL
#define COMMANDLETS_H_CL

#include <Utils/CommonIncludes.h>
namespace CaptainLucha
{
	class Commandlets
	{
	public:
		Commandlets();
		~Commandlets();

		bool ParseCommands(int argc, char** argv);
		void FireCommandlets();

		const std::vector<std::string>* GetCommandArguements(const std::string& command) const;

	protected:
		bool PreGameUseCommands();

	private:
		std::vector<std::pair<std::string, std::vector<std::string> > > m_commands;
	};
}

#endif
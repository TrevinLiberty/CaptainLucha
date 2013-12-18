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
 *	@file	CLConsole.h
 *	@brief	Console used for debugging purposes. Fires an Event to the EventSystem whenever input is pushed into the console by the user, 
				allowing for any object to capture user input from the console.
 *	@todo	Add alternate way to capture commands, besides just EventSystem?
 *  @see	EventSystem InputSystem
 *
/****************************************************************************/

#ifndef CONSOLE_H_CL
#define CONSOLE_H_CL

#include "CLConsole_Interface.h"
#include "Input/InputListener.h"
#include "Utils/UtilDebug.h"
#include "Utils/CommonIncludes.h"
#include "Threads/ThreadMutex.h"

namespace CaptainLucha
{
	class CLConsole : public CLConsole_Interface,  public InputListener
	{
	public:
		CLConsole();
		~CLConsole();

		/**
		 * @brief     Draws the console on the left half of the screen. Text scrolls button/up with the input field at the bottom left.
		 */
		virtual void Draw();

		/**
		 * @brief   virtual function from InputListener  
		 * @see		InputListener
		 */
		void KeyDown(int key);

		/**
		 * @brief     Adds help text that is displayed in the console when the user types 'help'
		 */
		void AddHelpInfo(const char* command, const char* info);

		/**
		 * @brief     Adds white text to the console
		 */
		void AddText(const char* text);

		/**
		 * @brief     Adds red text to the console
		 */
		void AddErrorText(const char* text);

		/**
		 * @brief     Adds green text to the console
		 */
		void AddSuccessText(const char* text);

		void Open() {m_isOpen = true;}
		void Close() {m_isOpen = false;}

		static const int CONSOLE_WIDTH = 750;
		static const int CONSOLE_HEIGHT = 920;

	protected:
		void ProcessInput(const std::string& input);
		void AddTextToHistory(bool isUserInput, const std::string& text);
		void CheckHistoryOverflow();

		std::vector<std::pair<const char*, const char*> > m_helpInfo;

		std::list<std::pair<bool, std::string> > m_history;

		std::string m_currentInput;

		ThreadMutex m_historyMutex;

		int m_cursorPos;
		int m_maxCharPerLine;

	private:
	};
}

#endif
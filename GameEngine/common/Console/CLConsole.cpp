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

#include "CLConsole.h"
#include "Input/InputSystem.h"
#include "Renderer/RendererUtils.h"
#include "Utils/Utils.h"
#include "EventSystem/NamedProperties.h"
#include "EventSystem/EventSystem.h"
#include "Utils/CLLogger.h"

#include <locale>

namespace CaptainLucha
{
    static const char* WHITE_COLOR     = "<#FFFFFFFF>";
    static const char* SUCCESS_COLOR   = "<#04B404FF>";
    static const char* ERROR_COLOR     = "<#DF0101FF>";
    static const char* DEFAULT_COLOR   = "<#C0C0C0FF>";
    static const char* HELP_INFO_COLOR = "<#AEB404FF>";

	CLConsole::CLConsole()
		: CLConsole_Interface(),
		InputListener(1000),
		m_cursorPos(0),
		m_maxCharPerLine(90)
	{

	}

	CLConsole::~CLConsole()
	{

	}

	void CLConsole::Draw()
	{
		if(m_isOpen)
		{
			SetUtilsColor(Color(0.25f, 0.25f, 0.25f, 0.5f));
			DrawBegin(CL_QUADS);
			{
				clVertex3(0.0f, 0.0f, 0.25f);
				clVertex3((float)CONSOLE_WIDTH, 0.0f, 0.25f);
				clVertex3((float)CONSOLE_WIDTH, (float)CONSOLE_HEIGHT, 0.25f);
				clVertex3(0.0f, (float)CONSOLE_HEIGHT, 0.25f);
			}
			DrawEnd();
			SetUtilsColor(Color::White);

			std::stringstream ss;
			std::string userInput(m_currentInput);
			userInput.insert(m_cursorPos, "|");
			ss << WHITE_COLOR << ">>> " << userInput;
			Draw2DDebugText(Vector2Df(0.0f, 5.0f), ss.str().c_str());

			float y = DEBUG_FONT_HEIGHT * (float)m_history.size() + DEBUG_FONT_HEIGHT;
			ss.str("");

			m_historyMutex.Lock();
			for(auto it = m_history.begin(); it != m_history.end(); ++it)
			{
				if((*it).first)
					ss << DEFAULT_COLOR << "<<< ";
				else
					ss << DEFAULT_COLOR << ">>> ";

				ss << (*it).second << "\n";
			}
			m_historyMutex.Unlock();

			Draw2DDebugText(Vector2Df(0.0f, y), ss.str().c_str());
		}
	}

	void CLConsole::KeyDown(int key)
	{
		if(key == GLFW_KEY_GRAVE_ACCENT)
		{
			m_isOpen = !m_isOpen;

// 			if(m_isOpen)
// 				InputSystem::GetInstance()->DisableInput();
// 			else
// 				InputSystem::GetInstance()->EnableInput();

			return;
		}

		if(m_isOpen)
		{
			if(key == GLFW_KEY_ENTER)
			{
				ProcessInput(m_currentInput);
				CLLogger::GetInstance()->Log(m_currentInput);
				m_cursorPos = 0;
				m_currentInput.clear();
			}
			else if(key == GLFW_KEY_BACKSPACE && !m_currentInput.empty())
			{
				if(m_cursorPos == (int)m_currentInput.size())
				{
					--m_cursorPos;
					m_currentInput.pop_back();
				}
				else
				{
					if(m_cursorPos > 0)
					{
						m_currentInput.erase(m_cursorPos - 1, 1);
						--m_cursorPos;
					}
				}
			}
			else if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_LEFT_CONTROL) && key == GLFW_KEY_V)
			{
				std::string pasteText = GetClipboardData_w();
				m_currentInput.append(pasteText);
				m_cursorPos = m_currentInput.size();
			}
			else if(InputSystem::GetInstance()->IsReadableKey(key)) //Space to Grave Accent, Readable Characters
			{
				char charKey = InputSystem::GetInstance()->GetActualKey(key);
				if(m_cursorPos == (int)m_currentInput.size())
				{
					++m_cursorPos;
					m_currentInput.push_back(charKey);
				}
				else
				{
					std::string c;
					c = charKey;

					m_currentInput.insert(m_cursorPos, c);
					++m_cursorPos;
				}
			}
			else if(key == GLFW_KEY_LEFT)
			{
				--m_cursorPos;
				m_cursorPos = max(0, m_cursorPos);
			}
			else if(key == GLFW_KEY_RIGHT)
			{
				++m_cursorPos;
				m_cursorPos = min(m_cursorPos, (int)m_currentInput.size());
			}
			else if(key == GLFW_KEY_DELETE)
			{
				if(m_cursorPos != (int)m_currentInput.size())
				{
					m_currentInput.erase(m_cursorPos, 1);
				}
			}
		}
	}

	void CLConsole::AddHelpInfo(const char* command, const char* info)
	{
		m_historyMutex.Lock();
		m_helpInfo.push_back(std::pair<const char*, const char*>(command, info));
		m_historyMutex.Unlock();
	}

	void CLConsole::AddText(const char* text)
	{
		AddTextToHistory(false, text);
	}

	void CLConsole::AddErrorText(const char* text)
	{
		std::stringstream ss;
		ss << ERROR_COLOR << text;
		AddTextToHistory(false, ss.str());
	}

	void CLConsole::AddSuccessText(const char* text)
	{
		std::stringstream ss;
		ss << SUCCESS_COLOR << text;
		AddTextToHistory(false, ss.str());
	}

	void CLConsole::ProcessInput(const std::string& input)
	{
		if(_strcmpi("help", input.c_str()) == 0)
		{
			AddTextToHistory(true, input);

            AddSuccessText("Help: Info");

			for(size_t i = 0; i < m_helpInfo.size(); ++i)
			{
				std::stringstream ss;
				ss << HELP_INFO_COLOR << m_helpInfo[i].first << ": " << m_helpInfo[i].second;
				AddTextToHistory(false, ss.str());
			}
		}
        else if(_strcmpi("help -e", input.c_str()) == 0)
        {
            AddTextToHistory(true, input);

            AddSuccessText("Help: Events");

            const SubscribersMap& subs = EventSystem::GetInstance()->GetSubscribers();
            for(auto it = subs.begin(); it != subs.end(); ++it)
            {
                std::stringstream ss;
                ss << HELP_INFO_COLOR << it->first;
                AddTextToHistory(false, ss.str());
            }
        }
		else
		{
			AddTextToHistory(true, input);

			NamedProperties np;
			std::stringstream ss;
			std::vector<std::string> tokens;
			TokenizeString(input, true, tokens);

			for(size_t i = 1; i < tokens.size(); ++i)
			{
				ss << "param" << i - 1;
				np.Set(ss.str(), tokens[i]);
				ss.str("");
			}

			FireEvent(tokens[0], np);
		}

		CheckHistoryOverflow();
	}

	void CLConsole::AddTextToHistory(bool isUserInput, const std::string& text)
	{
		m_historyMutex.Lock();
		if((int)text.size() > m_maxCharPerLine)
		{
			std::string historyText;
			historyText = text.substr(0, m_maxCharPerLine);
			m_history.push_back(std::pair<bool, std::string>(isUserInput, historyText));

			historyText = "    ";
			historyText += text.substr(m_maxCharPerLine, text.size());
			m_history.push_back(std::pair<bool, std::string>(isUserInput, historyText));
			CLLogger::GetInstance()->Log(historyText);
		}
		else
		{
			m_history.push_back(std::pair<bool, std::string>(isUserInput, text));
			CLLogger::GetInstance()->Log(text);
		}

		CheckHistoryOverflow();
		m_historyMutex.Unlock();
	}

	void CLConsole::CheckHistoryOverflow()
	{
		for(;;)
		{
			if(m_history.size() <= 50)
				break;

			m_history.pop_front();
		}
	}
}
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

#include "InputSystem.h"
#include "InputListener.h"
#include "Utils/DebugUIWidget.h"

#include <Renderer/RendererUtils.h>
#include <glfw3.h>
#include <iostream>
#include <locale>

using namespace CaptainLucha;

bool comp(InputListener* a, InputListener* b)
{
	return a->GetPriority() > b->GetPriority();
}

void InputSystem::RegisterListener(InputListener* listener)
{
	for(auto it = m_inputListeners.begin(); it != m_inputListeners.end(); ++it)
	{
		if(listener == *it)
			return;
	}
	m_inputListeners.push_back(listener);

	m_inputListeners.sort(comp);
}

void InputSystem::UnRegisterListener(InputListener* listener)
{
	for(auto it = m_inputListeners.begin(); it != m_inputListeners.end(); ++it)
	{
		if(listener == *it)
		{
			m_inputListeners.erase(it);
			break;
		}
	}

	m_inputListeners.sort(comp);
}

bool InputSystem::IsKeyDown(Key key) const
{
	return m_isEnabled && glfwGetKey(glfwGetCurrentContext(), key) == GLFW_PRESS;
}

bool InputSystem::IsKeyUp(Key key) const
{
	return m_isEnabled && glfwGetKey(glfwGetCurrentContext(), key) == GLFW_RELEASE;
}

void InputSystem::GetMousePos(Vector2Df& outPos)
{
	GetMousePos(outPos.x, outPos.y);
}

void InputSystem::GetMousePos(float& outX, float& outY)
{
	double x, y;
	glfwGetCursorPos(glfwGetCurrentContext(), &x, &y);
	outX = static_cast<float>(x);
	outY = static_cast<float>(y);
}

void InputSystem::SetMousePos(const Vector2Df& pos)
{
	SetMousePos(pos.x, pos.y);
}

void InputSystem::SetMousePos(float x, float y)
{
	glfwSetCursorPos(glfwGetCurrentContext(), (double)x, (double)(y));
}

char InputSystem::GetActualKey(int key)
{
	if(key == ' ')
		return ' ';

	if(IsKeyDown(GLFW_KEY_LEFT_SHIFT) || IsKeyDown(GLFW_KEY_RIGHT_SHIFT))
	{
		if((key >= 65 && key <= 90) || (key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_9))
			return (char)key;
		else
		{
			switch(key)
			{
			case GLFW_KEY_APOSTROPHE:
				return '"';
			case GLFW_KEY_COMMA:
				return '<';
			case GLFW_KEY_MINUS:
				return '_';
			case GLFW_KEY_PERIOD:
				return '>';
			case GLFW_KEY_SLASH:
				return '?';
			case GLFW_KEY_SEMICOLON:
				return ':';
			case GLFW_KEY_EQUAL:
				return '+';
			case GLFW_KEY_LEFT_BRACKET:
				return '{';
			case GLFW_KEY_BACKSLASH:
				return '|';
			case GLFW_KEY_RIGHT_BRACKET:
				return '}';
			case GLFW_KEY_GRAVE_ACCENT:
				return '~';
			case GLFW_KEY_0:
				return '!';
			case GLFW_KEY_1:
				return '@';
			case GLFW_KEY_2:
				return '#';
			case GLFW_KEY_3:
				return '$';
			case GLFW_KEY_4:
				return '%';
			case GLFW_KEY_5:
				return '^';
			case GLFW_KEY_6:
				return '&';
			case GLFW_KEY_7:
				return '*';
			case GLFW_KEY_8:
				return '(';
			case GLFW_KEY_9:
				return ')';
			}
		}
	}
	else if(IsReadableKey(key))
	{
		std::locale loc;
		return std::tolower((char)key, loc);
	}

	return ' ';
}

bool InputSystem::IsReadableKey(int key)
{
	return m_isEnabled && 
		((key >= GLFW_KEY_SPACE && key <= GLFW_KEY_GRAVE_ACCENT)
		|| 
		(key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_9));
}

bool InputSystem::IsMouseDown(int button)
{
	return m_isEnabled && glfwGetMouseButton(glfwGetCurrentContext(), button) == GLFW_PRESS;
}

void InputSystem::DeleteInstance()
{
	delete m_system;
	m_system = NULL;
}

////////////////////////////////////////////////////////////////////////
//			Protected
////////////////////////////////////////////////////////////////////////
InputSystem::InputSystem()
	: m_silenceKeyPress(false),
	  m_isEnabled(true),
	  m_debugMode(false),
	  m_hideMouseNotDebug(true),
	  m_debugToggleKey(GLFW_KEY_F1)
{
	glfwSetKeyCallback(glfwGetCurrentContext(), OnKeyboardEvent);
	glfwSetMouseButtonCallback(glfwGetCurrentContext(), OnMouseButton);
	glfwSetCursorPosCallback(glfwGetCurrentContext(), OnMouseMove);
}

InputSystem::~InputSystem()
{
	glfwSetKeyCallback(glfwGetCurrentContext(), NULL);
	glfwSetMouseButtonCallback(glfwGetCurrentContext(), NULL);
	glfwSetCursorPosCallback(glfwGetCurrentContext(), NULL);
}

void InputSystem::AlertAllListenersKeyboard(Key key, bool isDown)
{
	auto it = m_inputListeners.begin();

	for(it; it != m_inputListeners.end(); ++it)
	{
		if(isDown)
			(*it)->KeyDown(key);
		else
			(*it)->KeyUp(key);

		if(m_silenceKeyPress)
			break;
	}

	m_silenceKeyPress = false;
}

void InputSystem::AlertAllListenersMouse(int mouseButton, bool isDown)
{
	auto it = m_inputListeners.begin();

	for(it; it != m_inputListeners.end(); ++it)
	{
		(*it)->MouseClick(mouseButton, isDown);
	}
}

void InputSystem::AlertAllListenersMouseMove(float x, float y)
{
	auto it = m_inputListeners.begin();

	for(it; it != m_inputListeners.end(); ++it)
	{
		(*it)->MouseMove(x, y);
	}
}

//////////////////////////////////////////////////////////////////////////
//Private
//////////////////////////////////////////////////////////////////////////
InputSystem* InputSystem::m_system = NULL;

//////////////////////////////////////////////////////////////////////////
//	GLUT Callbacks
//////////////////////////////////////////////////////////////////////////
void CaptainLucha::OnKeyboardEvent(GLFWwindow* window, int key, int scancode, int state, int mods)
{
	UNUSED(window)
	UNUSED(scancode)
	UNUSED(mods)
	if(InputSystem::m_system->m_debugMode)
	{
		DebugUIKeyPress(key, state);
	}
	else
		InputSystem::m_system->AlertAllListenersKeyboard(key, state == GLFW_PRESS);

	if(InputSystem::m_system->m_debugToggleKey == key && state == GLFW_PRESS)
	{
		InputSystem::m_system->m_debugMode = !InputSystem::m_system->m_debugMode;
		if(InputSystem::m_system->m_debugMode)
			SetCursorHidden(false);
		else
			SetCursorHidden(InputSystem::m_system->m_hideMouseNotDebug);
	}
}

void CaptainLucha::OnMouseButton(GLFWwindow* window, int button, int state, int mods)
{
	UNUSED(window)
	UNUSED(mods)
	if(InputSystem::m_system->m_debugMode)
	{
		DebugUIMousePress(button, state);
	}
	else
		InputSystem::m_system->AlertAllListenersMouse(button, state == GLFW_PRESS);
}

void CaptainLucha::OnMouseMove(GLFWwindow* window, double x, double y)
{
	UNUSED(window)
	if(InputSystem::m_system->m_debugMode)
	{
		DebugUIMouseMove((int)x, (int)y);
	}
	else
		InputSystem::m_system->AlertAllListenersMouseMove((float)x, (float)y);
}
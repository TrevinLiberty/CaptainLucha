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
 *	@file	InputSystem.h
 *	@brief	
 *
/****************************************************************************/

#ifndef INPUT_SYSTEM_H_CL
#define INPUT_SYSTEM_H_CL

#include "Math/Vector2D.h"
#include "Utils/UtilMacros.h"
#include "Utils/UtilDebug.h"

#include <Utils/CommonIncludes.h>

#include <glfw3.h>

namespace CaptainLucha
{
	class InputListener;
	class InputSystem 
	{
	public:
		typedef int Key;

		static InputSystem* GetInstance()
		{
			if(m_system == NULL)
				m_system = new InputSystem();
			return m_system;
		}

		static void CreateInstance()
		{
			if(m_system == NULL)
				m_system = new InputSystem();
		}

		void RegisterListener(InputListener* listener);
		void UnRegisterListener(InputListener* listener);

		bool IsKeyDown(Key key) const;
		bool IsKeyUp(Key key) const;

		void GetMousePos(Vector2Df& outPos);
		void GetMousePos(float& outX, float& outY);

		void SetMousePos(const Vector2Df& pos);
		void SetMousePos(float x, float y);

		char GetActualKey(int key);
		bool IsReadableKey(int key);

		bool IsMouseDown(int button);

		void DeleteInstance();

		void SilenceInputForCurrentKey() {m_silenceKeyPress = true;}

		void EnableInput() {m_isEnabled = true;}
		void DisableInput() {m_isEnabled = false;}

		int GetDebugToggleKey() const {return m_debugToggleKey;}
		void SetDebugToggleKey(int val) {m_debugToggleKey = val;}

		bool IsDebugMode() const {return m_debugMode;}

		void SetShouldHideMouseOnNotDebug(bool val) {m_hideMouseNotDebug = val;}

	protected:
		InputSystem();
		~InputSystem();

		void AlertAllListenersKeyboard(Key key, bool isDown);
		void AlertAllListenersMouse(int mouseButton, bool isDown);
		void AlertAllListenersMouseMove(float x, float y);

	private:
		static InputSystem* m_system;

		bool m_silenceKeyPress;
		bool m_isEnabled;

		std::list<InputListener*> m_inputListeners;

		bool m_debugMode;
		bool m_hideMouseNotDebug;
		int m_debugToggleKey;

		PREVENT_COPYING(InputSystem);

		friend static void OnKeyboardEvent(GLFWwindow* window, int key, int scancode, int state, int mods);
		friend static void OnMouseButton(GLFWwindow* window, int button, int state, int mods);
		friend static void OnMouseMove(GLFWwindow* window, double x, double y);
	};
}

#endif
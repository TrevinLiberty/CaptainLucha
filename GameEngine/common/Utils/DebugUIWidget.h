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
 *	@file	DebugUIWidget.h
 *	@brief	
 *
/****************************************************************************/

#ifndef DEBUGUI_H_CL
#define DEBUGUI_H_CL

#include <Utils/Utils.h>
#include <Utils/TypedData.h>
#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	typedef void(__stdcall * ButtonCallback)(void* clientData);
	 
	class DebugUIWidgetImplementation;

	class DebugUIWidget
	{
	public:
		DebugUIWidget(const std::string& name);
		~DebugUIWidget();

		void SetPosition(float x, float y);
		void SetSize(float w, float h);

		void AddRWVariable(const std::string& name, CL_BaseTypes variableType, void* variable, const char* def = NULL);
		void AddROVariable(const std::string& name, CL_BaseTypes variableType, void* variable, const char* def = NULL);
		void AddButton(const std::string& name, ButtonCallback callback, void* clientData = NULL, const char* def = NULL);

	private:
		DebugUIWidgetImplementation* m_widget;
	};

	void InitWidgets();
	void DeInitWidgets();
	void DrawDebugUIWidgets();
	void SetScreenSizeForWidgets(int width, int height);
	
	//////////////////////////////////////////////////////////////////////////
	//	Input Events
	void DebugUIKeyPress(int key, int state);
	void DebugUIMousePress(int button, int state);
	void DebugUIMouseMove(int x, int y);
	void DebugUIMouseWheel(int pos);
}

#endif
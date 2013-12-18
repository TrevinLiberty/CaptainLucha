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

#include "DebugUIWidget.h"

#include <glfw3.h>
#include <AntTweakBar.h>
#include <Renderer/Color.h>
#include <Math/Quaternion.h>

namespace CaptainLucha
{
	class DebugUIWidgetImplementation
	{
	public:
		DebugUIWidgetImplementation(const std::string& name)
		{
			widget = TwNewBar(name.c_str());
		}

		~DebugUIWidgetImplementation()
		{
			TwDeleteBar(widget);
		}

		TwBar* widget;
	};

	DebugUIWidget::DebugUIWidget(const std::string& name)
	{
		m_widget = new DebugUIWidgetImplementation(name);
	}

	DebugUIWidget::~DebugUIWidget()
	{
		delete m_widget;
	}

	void DebugUIWidget::SetPosition(float x, float y)
	{
		float temp[] = {x, y};
		TwSetParam(m_widget->widget, NULL, "position", TW_PARAM_FLOAT, 2, &temp);
	}

	void DebugUIWidget::SetSize(float w, float h)
	{
		float temp[] = {w, h};
		TwSetParam(m_widget->widget, NULL, "size", TW_PARAM_FLOAT, 2, &temp);
	}

	TwType GetType(CL_BaseTypes variableType)
	{
		TwType type = TW_TYPE_UNDEF;
		switch(variableType)
		{
		case CL_BOOL:
			type = TW_TYPE_BOOLCPP;
			break;
		case CL_CHAR:
			type = TW_TYPE_CHAR;
			break;
		case CL_INT:
			type = TW_TYPE_INT32;
			break;
		case CL_UINT:
			type = TW_TYPE_UINT32;
			break;
		case CL_FLOAT:
			type = TW_TYPE_FLOAT;
			break;
		case CL_DOUBLE:
			type = TW_TYPE_DOUBLE;
			break;
		case CL_COLOR:
			type = TW_TYPE_COLOR4F;
			break;
		case CL_QUAT:
			type = TW_TYPE_QUAT4F;
			break;
		case CL_VECTOR:
			type = TW_TYPE_DIR3F;
			break;
		case CL_STRING:
			type = TW_TYPE_STDSTRING;
			break;
		}

		REQUIRES(type != TW_TYPE_UNDEF)

		return type;
	}

	void DebugUIWidget::AddRWVariable(const std::string& name, CL_BaseTypes variableType, void* variable, const char* def)
	{
		TwType type = GetType(variableType);
		TwAddVarRW(m_widget->widget, name.c_str(), type, variable, def);
	}

	void DebugUIWidget::AddROVariable(const std::string& name, CL_BaseTypes variableType, void* variable, const char* def)
	{
		TwType type = GetType(variableType);
		TwAddVarRO(m_widget->widget, name.c_str(), type, variable, def);
	}

	void DebugUIWidget::AddButton(const std::string& name, ButtonCallback callback, void* clientData, const char* def)
	{
		TwAddButton(m_widget->widget, name.c_str(), reinterpret_cast<TwButtonCallback>(callback), clientData, def);
	}

	//////////////////////////////////////////////////////////////////////////
	//	Private
	//////////////////////////////////////////////////////////////////////////
	void InitWidgets()
	{
		TwInit(TW_OPENGL, NULL);
	}

	void DeInitWidgets()
	{
		TwTerminate();
	}

	void DrawDebugUIWidgets()
	{
		TwDraw();
	}

	void SetScreenSizeForWidgets(int width, int height)
	{
		TwWindowSize(width, height);
	}

	//////////////////////////////////////////////////////////////////////////
	//	Input Events
	void DebugUIKeyPress(int key, int state)
	{
		switch(key)
		{
		case GLFW_KEY_ENTER:
			key = TW_KEY_RETURN;
			break;
		case GLFW_KEY_BACKSPACE:
			key = TW_KEY_BACKSPACE;
			break;
		}
		TwEventKeyGLFW(key, state);
		TwEventCharGLFW(key, state);
		TW_KEY_RETURN;
	}

	void DebugUIMousePress(int button, int state)
	{
		TwEventMouseButtonGLFW(button, state);
	}
	
	void DebugUIMouseMove(int x, int y)
	{
		TwEventMousePosGLFW(x, y);
	}
	
	void DebugUIMouseWheel(int pos)
	{
		TwEventMouseWheelGLFW(pos);
	}
}
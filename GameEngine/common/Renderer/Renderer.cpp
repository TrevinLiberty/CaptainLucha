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
 *	@file	Renderer.cpp
 *	@brief	
 *
/****************************************************************************/

#include "Renderer.h"

#include "RendererUtils.h"
#include "Objects/Object.h"

namespace CaptainLucha
{

	Renderer::Renderer()
		: m_debugDraw(false)
	{

	}

	Renderer::~Renderer()
	{

	}

	void Renderer::Draw()
	{

	}


	void Renderer::AddObjectToRender(Object* object)
	{
		REQUIRES(object)
		m_renderableObjects.push_back(object);
	}

	void Renderer::RemoveObject(Object* object)
	{
		for(auto it = m_renderableObjects.begin(); it != m_renderableObjects.end(); ++it)
		{
			if((*it) == object)
			{
				m_renderableObjects.erase(it);
				return;
			}
		}
	}

    void Renderer::AddDrawFunction(DrawFunction func, void* userData)
    {
        REQUIRES(func)
        m_drawFunctions.push_back(DrawPair(func, userData));
    }

    void Renderer::RemoveDrawFunction(DrawFunction func, void* userData)
    {
        REQUIRES(func)
        for(auto it = m_drawFunctions.begin(); it != m_drawFunctions.end(); ++it)
        {
            if(it->first == func && it->second == userData)
            {
                m_drawFunctions.erase(it);
                return;
            }
        }
    }

	void Renderer::DrawScene(GLProgram& program)
	{		
		for(size_t i = 0; i < m_renderableObjects.size(); ++i)
		{
			if(m_renderableObjects[i]->IsVisible())
			{
				program.SetUniform("ObjectID", m_renderableObjects[i]->GetID());
				m_renderableObjects[i]->Draw(program);
			}
		}

        if(m_debugDraw && m_debugDrawFunction)
            m_debugDrawFunction(program, m_debugDrawUserData);

        for(int i = 0; i < m_drawFunctions.size(); ++i)
        {
            m_drawFunctions[i].first(program, m_drawFunctions[i].second);
        }
	}
}
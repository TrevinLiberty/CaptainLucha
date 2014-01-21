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

#include "Game.h"

#include <Renderer/DeferredRenderer.h>
#include <Console/CLConsole.h>
#include <Input/InputSystem.h>

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	Game::Game(int argc, char** argv)
		: CLCore(argc, argv)
	{
        SetConsole(new CLConsole());
        SetRenderer(new DeferredRenderer());

		m_gameAbsClock = GetNewAbsoluteClockFromMaster();
		m_gameFrameClock = GetNewFrameClockFromMaster();

		m_gameWorld = new GameWorld(*this);
	}

	Game::~Game()
	{
		delete m_gameAbsClock;
		delete m_gameFrameClock;
	}

	void Game::Update()
	{
		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_ESCAPE))
			SetGameOver();

		m_gameWorld->Update();
	}

    void Game::PreDraw()
    {
        m_gameWorld->Draw();
    }

    void Game::PostDraw()
    {

    }

	//////////////////////////////////////////////////////////////////////////
	//	Private
	//////////////////////////////////////////////////////////////////////////
}
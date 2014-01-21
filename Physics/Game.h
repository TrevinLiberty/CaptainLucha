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



#ifndef GAME_H_CL
#define GAME_H_CL

#include <CLCore.h>
#include <Utils/CommonIncludes.h>
#include <GameWorld.h>

namespace CaptainLucha
{
	class GLProgram;

	class Game : public CLCore
	{
	public:
		Game(int argc, char** argv);
		virtual ~Game();

		Clock* GetAbsClock() { return m_gameAbsClock; }
		Clock* GetFrameClock() {return m_gameFrameClock;}

		Clock* GetNewAbsoluteClock() {return m_gameAbsClock->CreateNewClock();}
		Clock* GetNewFrameClock() {return m_gameFrameClock->CreateNewClock();}

	protected:
		void Update();

		void PreDraw();
        void PostDraw();

		void CreateNewSphere();

	private:
		Clock* m_gameAbsClock;
		Clock* m_gameFrameClock;

		GameWorld* m_gameWorld;
	};
}

#endif
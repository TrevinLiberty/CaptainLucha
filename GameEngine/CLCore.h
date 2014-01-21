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
 *	@mainpage
 *
 *
 *	@author Trevin Liberty
 *	@file	CLCore.h
 *	@brief	
 *
/****************************************************************************/

#ifndef GAME_ENGINE_H_CL
#define GAME_ENGINE_H_CL

#include "Input/InputListener.h"
#include "Utils/CLLogger.h"
#include "Renderer/RendererUtils.h"
#include "Renderer/Renderer.h"
#include "Renderer/Texture/GLTextureManager.h"
#include "Time/Clock.h"
#include "Time/FPSCalc.h"
#include "Utils/Commandlets.h"
#include "Console/CLConsole_Interface.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	extern int CL_VERSION_NUMBER;

	/**
	 * @brief     Initializes the core engine classes. This MUST be called before creating a CLCore instance.
	 * @param	  int argc
	 * @param	  char * * argv
	 */
	void InitCore(int argc, char** argv);

	class CLConsole_Interface;
	
	/**
	 * @attention Call InitCore before initializing.
	 */
	class CLCore
	{
	public:
		CLCore(int argc, char** argv);
		virtual ~CLCore();

		/**
		 * @brief     Desired number of frame per second.
		 * @see		  CLCore::SetEnableFrameLimiting
		 */
		static const int DESIRED_LIMIT_FRAMES_PER_SEC = 60;

		/**
		 * @brief     Starts the main loop. Returns when game is over.
		 * @see		  CLCore::SetGameOver
		 */
		void StartMainLoop();

		Renderer&	GetRenderer() {return *m_renderer;}
		void		SetRenderer(Renderer* newRenderer);
		Renderer*	SwitchRenderer(Renderer* newRenderer);

		/**
		 * @brief     Returns textures cached for use by the engine. 
		 *			  Textures that don't change per level should be loaded here.
		 * @return    GLTextureManager&
		 * @todo	  Add Way To Change
		 */
		GLTextureManager& GetEngineTextures() {return m_engineTextures;}

		/**
		 * @brief     Returns textures cached for use by levels.
		 *			  Allows deletion of textures when levels are changed.
		 * @return    GLTextureManager&
		 * @todo	  Add Way To Change
		 */
		GLTextureManager& GetLevelTextures() {return m_levelTextures;}

		/**
		 * @brief     Returns the engine's console.
		 * @return    CLConsole_Interface&
		 */
		CLConsole_Interface& GetConsole() {return *m_console;}

		/**
		 * @brief     Returns the engine's commandlets. This is loaded when CLCore is created.
		 * @return    Commandlets&
		 */
		Commandlets& GetCommandlets() {return m_commandlets;}

		/**
		 * @brief     Sets the game to quit on the next frame. Breaking the mainloop and CLCore::StartMainLoop to return.

		 */
		void SetGameOver() {m_gameOver = true;}
		bool IsGameOver() const {return m_gameOver;}

		/**
		 * @brief     GameOverEvent
		 * @param	  NamedProperties & np
		 */
		void GameOverEvent(NamedProperties& np) {UNUSED(np); SetGameOver();}

		/**
		 * @brief     ToggleProfilingInfo
		 * @param	  NamedProperties & np
		 */
		void ToggleProfilingInfo(NamedProperties& np);

		/**
		 * @brief     Returns clock that is updated, usually, by 1/60 second
		 * @return    Clock&
		 */
		const Clock& GetFrameClock() {return m_masterFrameClock;}

		/**
		 * @brief     Returns clock that is updated by absolute time
		 * @return    Clock&
		 */
		const Clock& GetAbsClock() {return m_masterAbsClock;}

		/**
		 * @brief     Enables or disables frame limiting.
		 * @param	  bool val

		 */
		void SetEnableFrameLimiting(bool val) {m_enableFrameLimiting = val;}

	protected:
		void Delete();

		/**
		 * @brief     Returns a new clock from CLCore::m_masterAbsClock
		 * @return    Clock*
		 */
		Clock* GetNewAbsoluteClockFromMaster();

		/**
		 * @brief     Returns a new clock from CLCore::m_masterFrameClock
		 * @return    Clock*
		 */
		Clock* GetNewFrameClockFromMaster();

		/**
		 * @brief     Sets the engine's console interface. Default is CLConsole. Changing to CLConsole_NULL will disable console output, input, and drawing.
		 * @param	  CLConsole_Interface * consoleInterface

		 */
		void SetConsole(CLConsole_Interface* consoleInterface);

		/**
		 * @brief     Update
		 * @pure

		 */
		virtual void Update() = 0;

		/**
		 * @brief     Draw
		 * @pure

		 */
		virtual void PreDraw() = 0;

		/**
		 * @brief     Draw
		 * @pure

		 */
		virtual void PostDraw() = 0;

		/**
		 * @brief     DrawHUD
		 * @pure

		 */
		virtual void DrawHUD() {};

	private:
		void MainLoop();

		void EngineUpdate();
		void EngineDraw();
		void EngineDrawHUD();

		void LogStartupInfo();

		bool m_gameOver;
		bool m_displayProfiler;
		bool m_drawMemoryVisualizer;
		bool m_outputMemoryInfo;
		bool m_serverMode;
		bool m_enableFrameLimiting;

		Clock m_masterAbsClock;
		Clock m_masterFrameClock;
		FPSCalc m_fpsCalc;
		Timer* m_memoryDebugTimer;

		CLConsole_Interface* m_console;
		Commandlets			 m_commandlets;

		Renderer*			 m_renderer;

		GLTextureManager	 m_engineTextures;
		GLTextureManager	 m_levelTextures;

		PREVENT_COPYING(CLCore)
	};
}

#endif
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

#include "CLCore.h"

#include <Utils/Memory/MemoryManager.h>

#include "Utils/Utils.h"
#include "EventSystem/EventSystem.h"
#include "Input/InputSystem.h"
#include "Collision/CollisionSystem.h"
#include "Time/Time.h"
#include "Time/ProfilingSystem.h"
#include "Time/ProfilingSection.h"
#include "Threads/JobManager.h"
#include "Utils/CLFileImporter.h"
#include "Utils/CLLogger.h"
#include "Utils/DebugUIWidget.h"

#include "Console/CLConsole_Interface.h"
#include "Console/CLConsole.h"

#include <glew.h>
#include <glfw3.h>
#include <math.h>

namespace CaptainLucha
{
	int CL_VERSION_NUMBER = 1000;

	void InitCore(int argc, char** argv)
	{
		UNUSED(argc) UNUSED(argv)
		ASSERT_MSG(InitRenderer(false), "Unable to Init GLFW!")

#ifndef _DEBUG
		FreeConsole();
#endif
		InitTime();
		InitWidgets();
		SetScreenSizeForWidgets(WINDOW_WIDTH, WINDOW_HEIGHT);
	}

	void GameOver()
	{
		JobManager::GetInstance()->WaitForThreadsAndDelete();
		JobManager::DeleteInstance();
		EventSystem::DeleteInstance();
		InputSystem::GetInstance()->DeleteInstance();
		ProfilingSystem::GetInstance()->DeleteInstance();
		DeleteRenderer();

		CLFileImporter fileImporter;
		fileImporter.DeleteStaticZipFromMemory();
		DeInitWidgets();
	}

	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	void CLCore::StartMainLoop()
	{
		MainLoop();
	}

	void CLCore::SetRenderer(Renderer* newRenderer)
	{
		REQUIRES(newRenderer)

		delete m_renderer;
		m_renderer = newRenderer;
	}

	Renderer* CLCore::SwitchRenderer(Renderer* newRenderer)
	{
		Renderer* oldRenderer = m_renderer;
		m_renderer = newRenderer;
		return oldRenderer;
	}

	void CLCore::ToggleProfilingInfo(NamedProperties& np)
	{
		UNUSED(np)
		m_displayProfiler = !m_displayProfiler;
		m_console->AddSuccessText("Profiling Info Toggled.");
	}

	Clock* CLCore::GetNewAbsoluteClockFromMaster()
	{
		return m_masterAbsClock.CreateNewClock();
	}

	Clock* CLCore::GetNewFrameClockFromMaster()
	{
		return m_masterFrameClock.CreateNewClock();
	}

	void CLCore::InjectConsole(CLConsole_Interface* consoleInterface)
	{
		REQUIRES(consoleInterface)
		delete m_console;
		m_console = consoleInterface;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	CLCore::CLCore(int argc, char** argv)
		: m_gameOver(false),
		  m_displayProfiler(false),
		  m_drawMemoryVisualizer(false),
		  m_outputMemoryInfo(false),
		  m_serverMode(false),
		  m_enableFrameLimiting(true),
		  m_console(new CLConsole()),
		  m_renderer(NULL)
	{
		UNUSED(argc) UNUSED(argv)

		m_commandlets.ParseCommands(argc, argv);

		if(m_commandlets.GetCommandArguements("server") != NULL)
		{
			m_serverMode = true;
			m_console->Open();
		}

		CLFileImporter fileImporter;
		fileImporter.LoadStaticZipFromDisk();
		fileImporter.SetLoadingType(CL_FOLDER_ZIP);
		
		EventSystem::CreateInstance();
		InputSystem::CreateInstance();
		JobManager::CreateInstance();
		ProfilingSystem::CreateInstace();
		CLLogger::CreateInstance();

		LogStartupInfo();

		m_console->AddHelpInfo("profile", "Displays Profiling Information.");

		EventSystem::GetInstance()->RegisterObjectForEvent("quit", *this, &CLCore::GameOverEvent);
		EventSystem::GetInstance()->RegisterObjectForEvent("profile", *this, &CLCore::ToggleProfilingInfo);

		m_masterAbsClock.SetMaxDT(1.0);
		m_masterFrameClock.SetMaxDT(1.0);

		m_memoryDebugTimer = m_masterFrameClock.CreateNewTimer();
		m_memoryDebugTimer->SetTimer(0.1f);
		m_memoryDebugTimer->StartTimer();
	}

	CLCore::~CLCore()
	{
		GameOver();
		delete m_memoryDebugTimer;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Private
	//////////////////////////////////////////////////////////////////////////
	void CLCore::MainLoop()
	{
		static const double DT = 1.0 / DESIRED_LIMIT_FRAMES_PER_SEC;
		const int MAX_FRAME_SKIP = 2;

		double frameTimeAccumulator = 0.0;
		double currentFrameTime = GetAbsoluteTimeSeconds();

		while(!m_gameOver && !glfwWindowShouldClose(glfwGetCurrentContext()))
		{
			ProfilingSection p("Frame");
			const double NEW_TIME = GetAbsoluteTimeSeconds();
			double frameTime = NEW_TIME - currentFrameTime;
			currentFrameTime = NEW_TIME;

			frameTimeAccumulator += frameTime;

			{
				ProfilingSection p("Network Update");
			}

			int numUpdates = 0;
			while(m_enableFrameLimiting && frameTimeAccumulator >= DT && numUpdates <= MAX_FRAME_SKIP)
			{
				ProfilingSection p("Update Loop");
				m_masterFrameClock.Update(DT);
				m_masterAbsClock.Update(frameTime);

				EngineUpdate();
				frameTimeAccumulator -= DT;
				++numUpdates;
			}

			{
				{
					ProfilingSection p("Draw Update1");
					EngineDraw();
				}
				{
					EngineDrawHUD();
					DrawDebugUIWidgets();
					ProfilingSection p("Draw Update2");
				}
				glfwSwapBuffers(glfwGetCurrentContext());
			}

			{
				ProfilingSection pp("Pool Events");
				glfwPollEvents();
			}

			ProfilingSystem::GetInstance()->InitNewFrame();

			std::stringstream ss;
			ss.precision(4);
			ss << std::fixed << "Game FPS: " << m_fpsCalc.GetFPS();

#ifdef NEW_NEW_LOL
			const int NUM_ALLOCATED_MB = static_cast<int>((MemoryManager::GetInstance().GetNumAllocatedBytes() / 1024.0f) / 1024.0f);
			const float NUM_FREE_MB = (MemoryManager::GetInstance().GetNumFreeBytes() / 1024.0f) / 1024.0f;
			ss << ", Aloc: " << NUM_ALLOCATED_MB << "MB, Used: " << NUM_ALLOCATED_MB - NUM_FREE_MB << "MB";
#endif

			SetWindowTitle(ss.str());
			m_fpsCalc.Update();
		}

		//TODO GAMEOVER
	}

	void CLCore::EngineUpdate()
	{
		JobManager::GetInstance()->Update();
		Update();

		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_F3) && m_memoryDebugTimer->IsTimeUp())
		{
			m_outputMemoryInfo = ! m_outputMemoryInfo;
			if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_LEFT_SHIFT))
				m_drawMemoryVisualizer = !m_drawMemoryVisualizer;

			m_memoryDebugTimer->StartTimer();
		}
	}

	void CLCore::EngineDraw()
	{
		g_MVPMatrix->LoadIdentity();
		HUDMode(false);
		glClearDepth( 1.f );
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		PreDraw();

		if(m_renderer)
			m_renderer->Draw();

		PostDraw();
	}

	void CLCore::EngineDrawHUD()
	{
		if(m_serverMode)
		{
			HUDMode(true);
			m_console->Draw();
			return;
		}

		ProfilingSection p("HUD Draw");
		DrawHUD();

		HUDMode(true);

		m_console->Draw();

#ifdef NEW_NEW_LOL
		if(m_drawMemoryVisualizer)
		{
			MemoryManager::GetInstance().DisplayVisualizer();
			MemoryManager::GetInstance().DisplayMemoryInfo();
		}
		else if(m_outputMemoryInfo)
			MemoryManager::GetInstance().DisplayMemoryInfo();
#endif

		if(m_displayProfiler)
			ProfilingSystem::GetInstance()->DebugDraw();
	}

	void CLCore::LogStartupInfo()
	{
		std::stringstream ss;
		
		CLLogger::GetInstance()->Log("//////////////////////////////////////////////////////////////////////////");
		CLLogger::GetInstance()->Log("CaptainLucha Engine Project");
		CLLogger::GetInstance()->Log("Author:           Trevin Liberty");

		ss << "Version:          " << CL_VERSION_NUMBER;
		CLLogger::GetInstance()->Log(ss.str());
		ss.str("");

		int minor, major;
		glGetIntegerv(GL_MINOR_VERSION, &minor);
		glGetIntegerv(GL_MAJOR_VERSION, &major);
		ss << "OpenGL Version:   " << minor << " " << major;
		CLLogger::GetInstance()->Log(ss.str());
		ss.str("");

		ss << "GLFW Version:     " << glfwGetVersionString();
		CLLogger::GetInstance()->Log(ss.str());
		ss.str("");

		ss << "Memory Allocated: " << (MemoryManager::GetInstance().GetNumAllocatedBytes() / 1024.0f) / 1024.0f << " MBytes";
		CLLogger::GetInstance()->Log(ss.str());
		ss.str("");

		CLFileImporter importer;
		ss << "File Load Order:  " << importer.GetLoadingTypeString();
		CLLogger::GetInstance()->Log(ss.str());
		ss.str("");

		CLLogger::GetInstance()->Log("//////////////////////////////////////////////////////////////////////////");
	}
}
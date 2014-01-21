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

#include "PlayerController.h"
#include "GameWorld.h"
#include "Game.h"

#include <Renderer/RendererUtils.h>
#include <glfw3.h>

namespace CaptainLucha
{
	PlayerController::PlayerController(GameWorld& gameWorld)
		: m_gameWorld(gameWorld),
		  m_selectedObject(NULL)
	{

	}

	PlayerController::~PlayerController()
	{

	}

	void PlayerController::Update()
	{
		m_camera.Update();

		if(m_selectedObject)
		{
			Vector3Df camDir(0.0f, 0.0f, -1.0f);
			camDir = m_camera.GetGLViewMatrix().GetTranspose().TransformRotation(camDir);

			m_selectedObject->GetRigidBody().SetPosition(m_camera.GetPosition() + camDir * 50.0f);
			m_selectedObject->GetRigidBody().SetAwakeState(true);
		}
	}

	void PlayerController::KeyDown(int key) 
	{
		if(key == GLFW_KEY_MINUS)
		{
			CollisionPhysicsObject* sphere = m_gameWorld.CreateNewSphere();
			sphere->SetPosition(m_camera.GetPosition());

			Vector3Df camDir(0.0f, 0.0f, -1.0f);
			camDir = m_camera.GetGLViewMatrix().GetTranspose().TransformRotation(camDir);

			sphere->GetRigidBody().AddVelocity(camDir * 300.0f);
		}
		else if(key == GLFW_KEY_EQUAL)
		{
			CollisionPhysicsObject* cube = m_gameWorld.CreateNewCube();
			cube->SetPosition(m_camera.GetPosition());
			cube->GetRigidBody().SetOrientation(Quaternion(Vector3Dd(0.0, 0.9, 0.1), rand()));

			Vector3Df camDir(0.0f, 0.0f, -1.0f);
			camDir = m_camera.GetGLViewMatrix().GetTranspose().TransformRotation(camDir);

			cube->GetRigidBody().AddVelocity(camDir * 300.0f);
		}
	}
}
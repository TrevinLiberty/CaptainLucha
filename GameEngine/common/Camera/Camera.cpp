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

#include "Camera.h"

#include <glew.h>
#include <Renderer/RendererUtils.h>

#include "Utils/Utils.h"
#include "Input/InputSystem.h"
#include "Math/Matrix3D.h"
#include "Input/InputSystem.h"

using namespace CaptainLucha;

//////////////////////////////////////////////////////////////////////////
//	Public
//////////////////////////////////////////////////////////////////////////

Camera::Camera()
	: m_rotation(0.0f, 0.0f, 0.0f),
	  m_position(0.0f, 0.0f, 0.0f),
	  m_drag(.1f),
	  CAMERA_IMPULSE(.050f),
	  m_enableKeyboard(true),
	  m_enableMouse(true)
{}

Camera::~Camera()
{

}

Vector3Df Camera::GetForwardDirection() const
{
    return (
        Matrix4Df((float)m_rotation.x, 1.0f, 0.0f, 0.0f)
      * Matrix4Df((float)m_rotation.y, 0.0f, 1.0f, 0.0f)
      ).TransformRotation(Vector3Df(0.0f, 0.0f, -1.0f));
}

Matrix4Df Camera::GetGLViewMatrix() const
{
	return Matrix4Df(-(float)m_position.x, -(float)m_position.y, -(float)m_position.z)
			* Matrix4Df(-(float)m_rotation.y, 0.0f, 1.0f, 0.0f)
			* Matrix4Df(-(float)m_rotation.x, 1.0f, 0.0f, 0.0f);
}

void Camera::ApplyImpulse(const Vector3Dd& impulse)
{
	Matrix3Dd rotationMatrix;
	rotationMatrix.MakeRotation(m_rotation.x, m_rotation.y, m_rotation.z); //Convert to radians
	m_velocity += rotationMatrix * impulse;
}

void Camera::Update()
{
	m_velocity -= m_velocity * m_drag;
	m_position += m_velocity;
	UpdateInput();
}

void Camera::UpdateInput()
{
	UpdateKeyboard();
}

void Camera::MouseMove(float x, float y)
{
	static const float cx = WINDOW_WIDTH * 0.5f;
	static const float cy = WINDOW_HEIGHT * 0.5f;

	const float MOUSE_YAW_POWER = .0008f;
	const float MOUSE_PITCH_POWER = .0008f;

	if(m_enableMouse)
	{
		float mouseDeltaX(x - cx);
		float mouseDeltaY(y - cy);

		// Update camera yaw.
		//
		m_rotation.y -= mouseDeltaX * MOUSE_YAW_POWER;

		// Update camera pitch.
		//
		m_rotation.x -= mouseDeltaY * MOUSE_PITCH_POWER;

		static const float MAX = 89.0f * Math<float>::PI / 180.0f;

		m_rotation.x = clamp(m_rotation.x, -MAX, MAX);

		InputSystem::GetInstance()->SetMousePos(cx, cy);
	}
}

//////////////////////////////////////////////////////////////////////////
//	Protected
//////////////////////////////////////////////////////////////////////////

void Camera::UpdateKeyboard()
{
	const float SPEED_INC_MULT = 50.0f;

	if(m_enableKeyboard)
	{
		Vector3Dd impulse;
		float speed = CAMERA_IMPULSE;

		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_LEFT_SHIFT))
			speed *= SPEED_INC_MULT;

		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_W))
		{
			impulse.z -= 1.0f;
		}
		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_S))
		{
			impulse.z += 1.0f;
		}

		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_A))
		{
			impulse.x -= 1.0;
		}
		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_D))
		{
			impulse.x += 1.0;
		}

		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_LEFT_CONTROL))
		{
			m_velocity.y -= speed * 2;
		}

		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_SPACE))
		{
			m_velocity.y += speed * 2;
		}

		impulse.Normalize();
		impulse *= speed;

		ApplyImpulse(impulse);
	}
}
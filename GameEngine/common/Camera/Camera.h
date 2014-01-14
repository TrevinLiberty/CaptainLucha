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
 *	@file	Camera.h
 *	@brief	
 *	@todo	Allow changing of speed
 *
/****************************************************************************/

#ifndef CAMERA_H_CL
#define CAMERA_H_CL

#include "Math/Vector3D.h"
#include "Math/Matrix4D.h"
#include "Utils/UtilMacros.h"
#include "Input/InputListener.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	/**
	* @class	Camera
	* @brief    Holds position and rotation for a player camera. Has functionality for movement. 
	*  @todo	Move camera movement to different file.
	*/
	class Camera : public InputListener
	{
	public:
		Camera();
		~Camera();

		void SetPosition(const Vector3Df& val) {m_position = val;}
		const Vector3Df& GetPosition() const {return m_position;}
		Vector3Df& GetPosition() {return m_position;}

		void SetVelocity(const Vector3Df& vel) {m_velocity = vel;}

		/**
		 * @brief     Returns (Yaw, Pitch, Roll)
		 * @return    const Vector3Df&
		 */
		const Vector3Df& GetRotation() const {return m_rotation;}

		/**
		 * @brief     Returns OpenGL ready view matrix
		 * @return    CaptainLucha::Matrix4Df
		 * @todo	  Create DX version
		 */
		Matrix4Df GetGLViewMatrix() const;

		/**
		 * @brief     Applies an impulse to the camera
		 * @param	  const Vector3Dd & impulse

		 */
		void ApplyImpulse(const Vector3Dd& impulse);

		/**
		 * @brief     Updates Position, Velocity, and input

		 */
		void Update();

		/**
		 * @brief     Updates Camera when mouse moves.
		 * @param	  float x
		 * @param	  float y

		 */
		void MouseMove(float x, float y);

		/**
		 * @brief     Enables or Disables keyboard input.
		 * @param	  bool t

		 */
		void SetEnableKeyboard(bool t) {m_enableKeyboard = t;}

		/**
		 * @brief     Enables or Disables mouse input.
		 * @param	  bool t

		 */
		void SetEnableMouse(bool t) {m_enableMouse = t;}

	protected:
		void UpdateKeyboard();
		void UpdateInput();

	private:
		Vector3Df m_rotation;
		Vector3Df m_position;
		Vector3Df m_velocity;

		float m_drag;

		bool m_enableKeyboard;
		bool m_enableMouse;

		const float CAMERA_IMPULSE;

		PREVENT_COPYING(Camera)
	};
}

#endif
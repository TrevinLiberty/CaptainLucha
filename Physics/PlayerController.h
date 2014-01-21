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

#ifndef PLAYERCONTROLLER_H_CL
#define PLAYERCONTROLLER_H_CL

#include <Camera/Camera.h>
#include <Input/InputListener.h>
#include <Objects/Object.h>

namespace CaptainLucha
{
	class GameWorld;
	class CollisionPhysicsObject;

	class PlayerController : public InputListener, public Object
	{
	public:
		PlayerController(GameWorld& gameWorld);
		~PlayerController();

		void Update();

		const Camera& GetCamera() {return m_camera;}

        void KeyDown(int key);

	protected:

	private:
		GameWorld& m_gameWorld;

		CollisionPhysicsObject* m_selectedObject;

		Camera m_camera;
	};
}

#endif
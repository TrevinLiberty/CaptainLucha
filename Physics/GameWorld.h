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



#ifndef GAMEWORLD_H_CL
#define GAMEWORLD_H_CL

#include <Collision/CollisionPhysicsObject.h>
#include <Physics/PhysicsSystem.h>
#include <Collision/StaticPlane.h>
#include <Renderer/Debug/DebugGrid.h>
#include <EventSystem/EventSystem.h>

#include "PlayerController.h"

namespace CaptainLucha
{
	class Light;
	class PhysicsObject;
	class Game;

	class GameWorld
	{
	public:
		GameWorld(Game& game);
		~GameWorld();

		void Draw();
		void Update();

		Game& GetCore() {return m_game;}

		CollisionPhysicsObject* CreateNewSphere();
		CollisionPhysicsObject* CreateNewCube();

		CollisionPhysicsObject* GetPhysicsObject(int id);

		int GetNumRigidBodies() const {return m_objects.size();}
		int GetNumLights() const {return m_lights.size();}

		void DebugCollision(NamedProperties& np);

	protected:
		void CreateLightPos(const Vector3Df& position);

	private:
		Game& m_game;

		PhysicsSystem m_physicsSystem;

		std::vector<CollisionPhysicsObject*> m_objects;
		std::vector<Light*> m_lights;

		StaticPlane m_ground;
		DebugGrid m_grid;

		PlayerController* m_player;

		PREVENT_COPYING(GameWorld)
	};
}

#endif
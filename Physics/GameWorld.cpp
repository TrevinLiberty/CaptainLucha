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

#include "GameWorld.h"
#include "Game.h"

#include <Collision/CollisionPhysicsObject.h>
#include <Collision/CollisionStatic.h>

#include <Physics/ForceGenerators/GravityGenerator.h>
#include <Physics/ForceGenerators/Spring.h>

#include <Renderer/Lights/Light.h>

namespace CaptainLucha
{
	static const float GROUND_SIZE = 500.0f;

	static const float WALL_HEIGHT = 50.0f;
	static const float WALL_WIDTH = 10.0f;

	GameWorld::GameWorld(Game& game)
		: m_game(game),
		  m_ground(Vector3Df(), Vector3Df(0.0f, 1.0f, 0.0f), Vector3Df(GROUND_SIZE, 0.0f, GROUND_SIZE)),
		  m_grid((int)GROUND_SIZE * 5, 10)
	{	
		m_ground.SetObjectColor(Color::White);

		m_physicsSystem.AddListener(&m_ground);

		m_game.GetRenderer().AddObjectToRender(&m_ground);
		m_game.GetRenderer().AddObjectToRender(&m_grid);

		m_player = new PlayerController(*this);
		m_player->SetPosition(0.0f, 100.0f, 0.0f);

		Light* ambientLight = m_game.GetRenderer().CreateNewAmbientLight();
		ambientLight->SetColor(Color::White);
		ambientLight->SetIntensity(0.01f);

		Vector3Df dir(-1.0f, -1.0f, 0.0f);
		dir.Normalize();

		Light_Directional* directionalLight = m_game.GetRenderer().CreateNewDirectionalLight();
		directionalLight->SetColor(Color::White);
		directionalLight->SetIntensity(0.1f);
		directionalLight->SetDirection(dir);

		GravityGenerator* gravity = new GravityGenerator();
		gravity->SetGravityForce(Vector3Df(0.0f, -25.0f, 0.0f));
		m_physicsSystem.AddNewWorldForceGenerator(gravity);

		CollisionStatic* walls[4];
		const float WALL_HALF_HEIGHT = WALL_HEIGHT * 0.5f;

		walls[0] = new CollisionStatic();
		walls[0]->SetPosition(Vector3Df(-GROUND_SIZE, WALL_HALF_HEIGHT, 0.0f));
		walls[0]->InitCuboid(Vector3Df(-GROUND_SIZE, WALL_HALF_HEIGHT, 0.0f), Quaternion(), Vector3Df(WALL_WIDTH, WALL_HEIGHT, GROUND_SIZE));
		m_game.GetRenderer().AddObjectToRender(walls[0]);
		walls[0]->SetObjectColor(Color::White);
		m_physicsSystem.AddListener(walls[0]);

		walls[1] = new CollisionStatic();
		walls[1]->SetPosition(Vector3Df(GROUND_SIZE, WALL_HALF_HEIGHT, 0.0f));
		walls[1]->InitCuboid(Vector3Df(GROUND_SIZE, WALL_HALF_HEIGHT, 0.0f), Quaternion(), Vector3Df(WALL_WIDTH, WALL_HEIGHT, GROUND_SIZE));
		m_game.GetRenderer().AddObjectToRender(walls[1]);
		walls[1]->SetObjectColor(Color::White);
		m_physicsSystem.AddListener(walls[1]);

		walls[2] = new CollisionStatic();
		walls[2]->SetPosition(Vector3Df(0.0f, WALL_HALF_HEIGHT, -GROUND_SIZE));
		walls[2]->InitCuboid(Vector3Df(0.0f, WALL_HALF_HEIGHT, -GROUND_SIZE), Quaternion(), Vector3Df(GROUND_SIZE, WALL_HEIGHT, WALL_WIDTH));
		m_game.GetRenderer().AddObjectToRender(walls[2]);
		walls[2]->SetObjectColor(Color::White);
		m_physicsSystem.AddListener(walls[2]);

		walls[3] = new CollisionStatic();
		walls[3]->SetPosition(Vector3Df(0.0f, WALL_HALF_HEIGHT, GROUND_SIZE));
		walls[3]->InitCuboid(Vector3Df(0.0f, WALL_HALF_HEIGHT, GROUND_SIZE), Quaternion(), Vector3Df(GROUND_SIZE, WALL_HEIGHT, WALL_WIDTH));
		m_game.GetRenderer().AddObjectToRender(walls[3]);
		walls[3]->SetObjectColor(Color::White);
		m_physicsSystem.AddListener(walls[3]);

		CreateLightPos(Vector3Df(0.0f, 0.0f, 350.0f));
		CreateLightPos(Vector3Df(0.0f, 0.0f, -350.0f));
		CreateLightPos(Vector3Df(350.0f, 0.0f, 0.0f));
		CreateLightPos(Vector3Df(-350.0f, 0.0f, 0.0f));

		//for(int i = 0; i < 4; ++i)
		//{
		//	for(int j = 0; j < 4; ++j)
		//	{
		//		for(int k = 0; k < 4; ++k)
		//		{
		//			CollisionPhysicsObject* object = NULL;

		//			if(i *k % 2 == 0)
		//				object = CreateNewCube();
		//			else
		//				object = CreateNewSphere();

		//			object->SetPosition(Vector3Df(i * 50.0f + RandInRange(-10.0f, 10.0f), j * 50.0f + 100.0f, k * 50.0f));
		//			object->GetRigidBody().AddVelocity(Vector3Dd(0.0, -1.0, 0.0));
		//			object->GetRigidBody().SetAwakeState(true);
		//		}
		//	}
		//}

		m_game.GetRenderer().AddObjectToRender(&m_physicsSystem);
		m_physicsSystem.SetVisible(false);

		EventSystem::GetInstance()->RegisterObjectForEvent("debugcollision", *this, &GameWorld::DebugCollision);
	}

	GameWorld::~GameWorld()
	{

	}

	void GameWorld::Draw()
	{
		m_game.GetRenderer().SetCameraPos(m_player->GetCamera().GetPosition());
		m_game.GetRenderer().SetViewMatrix(m_player->GetCamera().GetGLViewMatrix());
	}

	void GameWorld::Update()
	{
		for(size_t i = 0; i < m_objects.size(); ++i)
		{
			m_lights[i]->SetPosition(m_objects[i]->GetPosition());
		}

		m_physicsSystem.Update(1.0 / 60.0);
		m_player->Update();
	}

	CollisionPhysicsObject* GameWorld::CreateNewSphere()
	{
		CollisionPhysicsObject* newSphere = new CollisionPhysicsObject();
		newSphere->InitSphere(5.0f, 10.0f);

		newSphere->GetRigidBody().SetAngularDamp(0.7f);
		newSphere->GetRigidBody().SetLinearDamp(0.7f);
		newSphere->GetRigidBody().CalculatedDerivedData();

		Light* newLight = m_game.GetRenderer().CreateNewPointLight();
		newLight->SetIntensity(1.0f);
		Vector3Df color(RandInRange(0.0f, 1.0f), RandInRange(0.0f, 1.0f), RandInRange(0.0f, 1.0f)); color.Normalize();
		newLight->SetColor(Color(color.x, color.y, color.z));
		newSphere->SetObjectColor(Color(color.x, color.y, color.z));
		newLight->SetRadius(25.0f);

		m_physicsSystem.AddPhysicsObject(newSphere);
		m_game.GetRenderer().AddObjectToRender(newSphere);

		m_objects.push_back(newSphere);
		m_lights.push_back(newLight);

		return newSphere;
	}

	CollisionPhysicsObject* GameWorld::CreateNewCube()
	{
		CollisionPhysicsObject* newCube = new CollisionPhysicsObject();
		newCube->InitCuboid(Vector3Df(5.0f, 5.0f, 5.0f), 50.0f);

		newCube->GetRigidBody().SetAngularDamp(0.7f);
		newCube->GetRigidBody().SetLinearDamp(0.7f);
		newCube->GetRigidBody().CalculatedDerivedData();

		Light* newLight = m_game.GetRenderer().CreateNewPointLight();
		newLight->SetIntensity(1.0f);
		Vector3Df color(RandInRange(0.0f, 1.0f), RandInRange(0.0f, 1.0f), RandInRange(0.0f, 1.0f)); color.Normalize();
		newLight->SetColor(Color(color.x, color.y, color.z));
		newCube->SetObjectColor(Color(color.x, color.y, color.z));
		newLight->SetRadius(25.0f);

		m_physicsSystem.AddPhysicsObject(newCube);
		m_game.GetRenderer().AddObjectToRender(newCube);

		m_objects.push_back(newCube);
		m_lights.push_back(newLight);

		return newCube;
	}

	CollisionPhysicsObject* GameWorld::GetPhysicsObject(int id)
	{
		for(size_t i = 0; i < m_objects.size(); ++i)
			if(m_objects[i]->GetID() == id)
				return m_objects[i];
		return NULL;
	}

	bool tempToggle = false;
	void GameWorld::DebugCollision(NamedProperties& np)
	{
		tempToggle = !tempToggle;
		m_physicsSystem.SetVisible(tempToggle);
	}

	void GameWorld::CreateLightPos(const Vector3Df& position)
	{
		const float POLE_HEIGHT = 75.0f;
		const float POLE_WIDTH = 2.0f;
		const float HANG_LENGTH = 55.0f;
		const float LIGHT_SIDE_DROPDOWN_LENGTH = 5.0f; 

		CollisionStatic* lightPost = new CollisionStatic();
		lightPost->InitCuboid(
			position + Vector3Df(0.0f, POLE_HEIGHT, 0.0f), 
			Quaternion(), 
			Vector3Df(POLE_WIDTH, POLE_HEIGHT, POLE_WIDTH));
		m_game.GetRenderer().AddObjectToRender(lightPost);
		m_physicsSystem.AddListener(lightPost);

		CollisionStatic* lightPostArm = new CollisionStatic();
		lightPostArm->InitCuboid(
			position + Vector3Df(0.0f, POLE_HEIGHT+POLE_HEIGHT, 0.0f), 
			Quaternion(), 
			Vector3Df(HANG_LENGTH, POLE_WIDTH, POLE_WIDTH));
		m_game.GetRenderer().AddObjectToRender(lightPostArm);
		m_physicsSystem.AddListener(lightPostArm);

		CollisionStatic* lightPosDropDown1 = new CollisionStatic();
		lightPosDropDown1->InitCuboid(
			position + Vector3Df(HANG_LENGTH - POLE_WIDTH, POLE_HEIGHT+POLE_HEIGHT-LIGHT_SIDE_DROPDOWN_LENGTH, 0.0f), 
			Quaternion(), 
			Vector3Df(POLE_WIDTH, LIGHT_SIDE_DROPDOWN_LENGTH, POLE_WIDTH));
		m_game.GetRenderer().AddObjectToRender(lightPosDropDown1);
		m_physicsSystem.AddListener(lightPosDropDown1);

		CollisionStatic* lightPosDropDown2 = new CollisionStatic();
		lightPosDropDown2->InitCuboid(
			position + Vector3Df(-HANG_LENGTH + POLE_WIDTH, POLE_HEIGHT+POLE_HEIGHT-LIGHT_SIDE_DROPDOWN_LENGTH, 0.0f), 
			Quaternion(), 
			Vector3Df(POLE_WIDTH, LIGHT_SIDE_DROPDOWN_LENGTH, POLE_WIDTH));
		m_game.GetRenderer().AddObjectToRender(lightPosDropDown2);
		m_physicsSystem.AddListener(lightPosDropDown2);

		CollisionPhysicsObject* sphere1 = CreateNewSphere();
		sphere1->SetPosition(position + Vector3Df(-HANG_LENGTH + POLE_WIDTH, POLE_HEIGHT, 0.0f));
		sphere1->GetRigidBody().SetAwakeState(true);
		m_lights.back()->SetRadius(250.0f);

		SpringRigidBodyToPoint* newSpring1 = new SpringRigidBodyToPoint(&sphere1->GetRigidBody(), Vector3Df(), position + Vector3Df(-HANG_LENGTH + POLE_WIDTH, POLE_HEIGHT+POLE_HEIGHT, 0.0f));
		newSpring1->SetSpringRestLength(POLE_HEIGHT);
		newSpring1->SetSpringConstant(10000.0f);
		m_physicsSystem.AddNewCustomForceGenerator(newSpring1);	

		CollisionPhysicsObject* sphere2 = CreateNewSphere();
		sphere2->SetPosition(position + Vector3Df(HANG_LENGTH - POLE_WIDTH, POLE_HEIGHT, 0.0f));
		sphere2->GetRigidBody().SetAwakeState(true);
		m_lights.back()->SetRadius(250.0f);

		SpringRigidBodyToPoint* newSpring2 = new SpringRigidBodyToPoint(&sphere2->GetRigidBody(), Vector3Df(), position + Vector3Df(HANG_LENGTH - POLE_WIDTH, POLE_HEIGHT+POLE_HEIGHT, 0.0f));
		newSpring2->SetSpringRestLength(POLE_HEIGHT);
		newSpring2->SetSpringConstant(10000.0f);
		m_physicsSystem.AddNewCustomForceGenerator(newSpring2);	

		m_game.GetRenderer().AddObjectToRender(newSpring1);
		m_game.GetRenderer().AddObjectToRender(newSpring2);
	}
}
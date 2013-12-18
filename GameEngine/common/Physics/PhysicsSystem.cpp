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

#include "PhysicsSystem.h"
#include "Time/ProfilingSection.h"
#include "ContactResolver.h"
#include "Renderer/RendererUtils.h"

#include "Collision/CollisionPhysicsObject.h"
#include "Collision/CollisionMovable.h"
#include "Collision/CollisionStatic.h"

namespace CaptainLucha
{
	PhysicsSystem::PhysicsSystem()
	{

	}

	PhysicsSystem::~PhysicsSystem()
	{

	}

	//std::vector<Vector3D<Real>> pos;
	//std::vector<Vector3D<Real>> normal;
	void PhysicsSystem::Draw(GLProgram& glProgram)
	{
		//SetColor(Color::White);
		SetGLProgram(&glProgram);
		//for(int i = 0; i < pos.size(); ++i)
		//	DrawDebugArrow(pos[i], pos[i] + normal[i] * 10.0f);
		//	DrawDebugPlus(pos[i], 2.5f);
		//pos.clear();

		m_collisionSystem.DebugDraw();
		SetGLProgram(NULL);
		SetColor(Color::White);
	}

	void PhysicsSystem::AddPhysicsObject(CollisionPhysicsObject* object)
	{
		REQUIRES(object)
		REQUIRES(object->IsInit())
		m_objects.push_back(object);
		m_collisionSystem.AddListener(object);
	}

	void PhysicsSystem::RemovePhysicsObject(CollisionPhysicsObject* object)
	{
		REQUIRES(object)
		for(auto it = m_objects.begin(); it != m_objects.end();)
		{
			if((*it) == object)
				it = m_objects.erase(it);
			else
				++it;
		}

		m_collisionSystem.RemoveListener(object);
	}

	void PhysicsSystem::AddListener(CollisionListener* listener)
	{
		REQUIRES(listener)
		m_collisionSystem.AddListener(listener);
	}

	void PhysicsSystem::RemoveListener(CollisionListener* listener)
	{
		REQUIRES(listener)
		m_collisionSystem.RemoveListener(listener);
	}

	void PhysicsSystem::AddNewWorldForceGenerator(WorldForceGenerator* generator)
	{
		REQUIRES(generator)
		m_worldForceGenerators.push_back(generator);
	}

	void PhysicsSystem::RemoveWorldForceGenerator(WorldForceGenerator* generator)
	{
		REQUIRES(generator)
		for(auto it = m_worldForceGenerators.begin(); it != m_worldForceGenerators.end();)
		{
			if((*it) == generator)
				it = m_worldForceGenerators.erase(it);
			else
				++it;
		}
	}

	void PhysicsSystem::AddNewCustomForceGenerator(CustomForceGenerator* generator)
	{
		REQUIRES(generator)
		m_customForceGenerators.push_back(generator);
	}

	void PhysicsSystem::RemoveCustomForceGenerator(CustomForceGenerator* generator)
	{
		REQUIRES(generator)
		for(auto it = m_customForceGenerators.begin(); it != m_customForceGenerators.end();)
		{
			if((*it) == generator)
				it = m_customForceGenerators.erase(it);
			else
				++it;
		}
	}

	void PhysicsSystem::Update(double deltaTime)
	{
		const int NUM_ITERATIONS = 16;
		for(int i = 0; i < NUM_ITERATIONS; ++i)
		{
			const double DT = deltaTime / NUM_ITERATIONS;
			//////////////////////////////////////////////////////////////////////////
			//	Add Forces
			{
				ProfilingSection p("Force Update");
				for(auto forceit = m_customForceGenerators.begin(); forceit != m_customForceGenerators.end(); ++forceit)
					(*forceit)->UpdateForce(DT);

				for(auto forceit = m_worldForceGenerators.begin(); forceit != m_worldForceGenerators.end(); ++forceit)
				{
					for(auto objectit = m_objects.begin(); objectit != m_objects.end(); ++objectit)
						(*forceit)->UpdateForce(&(*objectit)->GetRigidBody(), DT);
					(*forceit)->FrameUpdate(DT);
				}
			}

			////////////////////////////////////////////////////////////////////////
			//	Update Position
			{
				ProfilingSection p("RB Update");
				for(auto it = m_objects.begin(); it != m_objects.end(); ++it)
				{
					(*it)->Update(DT);
				}
			}

			//////////////////////////////////////////////////////////////////////////
			//	Update Collisions
			{
				ProfilingSection p("Collision Update");
				m_collisionSystem.Update();
			}

			//////////////////////////////////////////////////////////////////////////
			//	Resolve Collisions
			{
				ProfilingSection p("Contact Update");
				CollisionPairs& collisions = m_collisionSystem.GetCollisions();
				for(size_t i = 0; i < collisions.size(); ++i)
				{
					CollisionListener* c1 = collisions[i].first;
					CollisionListener* c2 = collisions[i].second;

					CollisionPhysicsObject* colBody1 = NULL;
					CollisionPhysicsObject* colBody2 = NULL;
					RigidBody* body1 = NULL;
					RigidBody* body2 = NULL;

					if(c1->GetType() == CL_PHYSICS_OBJECT)
					{
						colBody1 = reinterpret_cast<CollisionPhysicsObject*>(c1);
						body1 = &colBody1->GetRigidBody();
					}
					if(c2->GetType() == CL_PHYSICS_OBJECT)
					{
						colBody2 = reinterpret_cast<CollisionPhysicsObject*>(c2);
						body2 = &colBody2->GetRigidBody();
					}

					if(colBody1 == NULL && colBody2 == NULL)
						continue;

					ContactResolver resolver(body1, c1->GetPrimitve(), body2, c2->GetPrimitve());
					resolver.GenerateContacts();
					
					//for(int j = 0; j < resolver.m_contacts.size(); ++j)
					//{
					//	pos.push_back(resolver.m_contacts[j].m_contactPoint);
					//	normal.push_back(resolver.m_contacts[j].m_contactNormal);
					//}

					resolver.ResolveContacts();
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////
////	Add Forces
//for(auto forceit = m_customForceGenerators.begin(); forceit != m_customForceGenerators.end(); ++forceit)
//	(*forceit)->UpdateForce(DT);
//
//for(auto forceit = m_worldForceGenerators.begin(); forceit != m_worldForceGenerators.end(); ++forceit)
//{
//	for(auto objectit = m_objects.begin(); objectit != m_objects.end(); ++objectit)
//		(*forceit)->UpdateForce(&(*objectit)->GetRigidBody(), DT);
//	(*forceit)->FrameUpdate(DT);
//}
//
////////////////////////////////////////////////////////////////////////////
////	Update Collisions
//m_collisionSystem.Update();
//
////////////////////////////////////////////////////////////////////////////
////	Update Velocity
//for(auto it = m_objects.begin(); it != m_objects.end(); ++it)
//{
//	(*it)->GetRigidBody().IntegrateVelocity(DT);
//}
//
////////////////////////////////////////////////////////////////////////////
////	Resolve Collisions
//CollisionPairs& collisions = m_collisionSystem.GetCollisions();
//for(size_t i = 0; i < collisions.size(); ++i)
//{
//	CollisionListener* c1 = collisions[i].first;
//	CollisionListener* c2 = collisions[i].second;
//
//	ContactResolver resolver(c1->GetRigidBody(), c1->GetPrimitve(), c2->GetRigidBody(), c2->GetPrimitve());
//	resolver.GenerateContacts();
//	resolver.ResolveContacts();
//}
//
////////////////////////////////////////////////////////////////////////////
////	Update Position
//for(auto it = m_objects.begin(); it != m_objects.end(); ++it)
//{
//	(*it)->GetRigidBody().IntegratePosition(DT);
//	(*it)->Update(DT);
//}
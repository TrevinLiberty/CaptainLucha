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
 *	@file	PhysicsSystem.h
 *	@brief	
 *
/****************************************************************************/

#ifndef PHYSICSSYSTEM_H_CL
#define PHYSICSSYSTEM_H_CL

#include "Collision/CollisionPhysicsObject.h"
#include "ForceGenerators/ForceGenerator.h"
#include "ForceGenerators/Cable.h"
#include "Collision/CollisionSystem.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class PhysicsSystem : public Object
	{
	public:
		PhysicsSystem();
		~PhysicsSystem();

		void Draw(GLProgram& glProgram);

		void AddPhysicsObject(CollisionPhysicsObject* object);
		void RemovePhysicsObject(CollisionPhysicsObject* object);

		void AddListener(CollisionListener* listener);
		void RemoveListener(CollisionListener* listener);

		void AddNewWorldForceGenerator(WorldForceGenerator* generator);
		void RemoveWorldForceGenerator(WorldForceGenerator* generator);

		void AddNewCustomForceGenerator(CustomForceGenerator* generator);
		void RemoveCustomForceGenerator(CustomForceGenerator* generator);

		void Update(double DT);

	protected:

	private:
		CollisionSystem m_collisionSystem;

		std::vector<CollisionPhysicsObject*> m_objects;
		std::vector<WorldForceGenerator*> m_worldForceGenerators;
		std::vector<CustomForceGenerator*> m_customForceGenerators;
	};
}

#endif
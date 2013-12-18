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
 *	@file	CollisionSystem.h
 *	@brief	
 *
/****************************************************************************/

#ifndef COLLISIONSYSTEM_H_CL
#define COLLISIONSYSTEM_H_CL

#include <Utils/CommonIncludes.h>
#include "CollisionFunctions.h"

namespace CaptainLucha
{
	struct CollisionNode;
	class CollisionListener;

	typedef std::vector<std::pair<CollisionListener*, CollisionListener*> > CollisionPairs;

	class CollisionSystem
	{
	public:
		CollisionSystem();
		~CollisionSystem();

		/**
		 * @brief     Builds the dynamic tree using all dynamic listeners and runs every dynamic object through the static and dynamic tree.
		 */
		void Update();

		/**
		 * @brief     Draws the AABB bounding volume tree using CL_Lines.
		 */
		void DebugDraw();

		/**
		 * @brief     Add the listener to the system. If the listener is static, it will be placed in the static BV tree.
		 */
		void AddListener(CollisionListener* listener);
		void RemoveListener(CollisionListener* listener);

		/**
		 * @brief     Retruns the collision pairs since the last update. Used for debuging.
		 */
		CollisionPairs& GetCollisions() {return m_collisions;}

	protected:
		void AddNewListeners();

		void AddListenerToTree(CollisionListener* listener, CollisionNode* currentNode);

		void AddStaticListener(CollisionListener* listener);
		void AddDynamicListener(CollisionListener* listener);

		void DetectCollisionsP(CollisionListener* listener, CollisionNode* currentNode);

		void DetectDynamicToStaticCollisions();
		void DetectDynamicToDynamicCollisions();

		void DebugDrawP(CollisionNode* currentNode, bool staticDraw);

		void DestroyTree(CollisionNode* currentNode);

		void AddCollision(CollisionListener* a, CollisionListener* b);

		void AlertListeners(CollisionListener* a, CollisionListener* b);

	private:
		std::vector<CollisionListener*> m_newListeners;
		std::vector<CollisionListener*> m_staticListeners;
		std::vector<CollisionListener*> m_dynamicListeners;
		
		CollisionPairs m_collisions;

		CollisionNode* m_staticRootNode;
		CollisionNode* m_dynamicRootNode;
	};
}

#endif
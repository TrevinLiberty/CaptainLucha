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

#include "CollisionSystem.h"
#include "CollisionListener.h"
#include "Renderer/Geometry/Mesh.h"
#include "Renderer/RendererUtils.h"
#include "BoundingVolumes/AABoundingBox.h"
#include "Math/Math.h"
#include "Physics/RigidBody.h"

namespace CaptainLucha
{
	struct CollisionNode
	{
		AABoundingBox m_aabb;

		Real m_cost;
		CollisionListener* m_listener;

		CollisionNode* m_left;
		CollisionNode* m_right;

		CollisionNode()
			: m_cost(0.0f), m_listener(NULL), m_left(NULL), m_right(NULL) {}

		AABoundingBox GetAABB()
		{
			if(m_listener)
				return m_listener->GetBV()->ConvertToAABB();
			else
				return m_aabb;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	void RemoveDuplicates(std::vector<std::pair<CollisionListener*, CollisionListener*> >& collisions)
	{
		for(auto it = collisions.begin(); it != collisions.end();)
		{
			auto inIt = it + 1;
			bool removed = false;

			for(;inIt != collisions.end(); ++inIt)
			{
				if(inIt->first == it->second && inIt->second == it->first)
				{
					it = collisions.erase(it);
					removed = true;
					break;
				}
			}

			if(!removed)
				++it;
		}
	}

	void CollisionSystem::Update()
	{
		m_collisions.clear();
		DestroyTree(m_dynamicRootNode);
		delete m_dynamicRootNode;
		m_dynamicRootNode = NULL;

		for(size_t i = 0; i < m_dynamicListeners.size(); ++i)
		{
			AddDynamicListener(m_dynamicListeners[i]);
		}

 		AddNewListeners();

		DetectDynamicToStaticCollisions();
		DetectDynamicToDynamicCollisions();

		RemoveDuplicates(m_collisions);
	}

	void CollisionSystem::DebugDrawP(CollisionNode* currentNode, bool staticDraw)
	{
		UNUSED(currentNode)
		if(!currentNode)
			return;

		if(currentNode->m_listener)
		{
			currentNode->m_listener->GetBV()->DebugDraw();
		}
		else
		{
			currentNode->m_aabb.DebugDraw();
		}

		DebugDrawP(currentNode->m_left, staticDraw);
		DebugDrawP(currentNode->m_right, staticDraw);
	}

	void CollisionSystem::DestroyTree(CollisionNode* currentNode)
	{
		if(!currentNode)
			return;

		DestroyTree(currentNode->m_left);
		DestroyTree(currentNode->m_right);

		delete currentNode->m_left;
		delete currentNode->m_right;

		currentNode->m_left = NULL;
		currentNode->m_right = NULL;
	}

	void CollisionSystem::DebugDraw()
	{
		//DebugDrawP(m_staticRootNode, true);
		DebugDrawP(m_dynamicRootNode, false);
	}

	void CollisionSystem::AddListener(CollisionListener* listener)
	{
		REQUIRES(listener)

		listener->m_collisionSystem = this;
		m_newListeners.push_back(listener);
	}

	void CollisionSystem::RemoveListener(CollisionListener* listener)
	{
		UNUSED(listener)
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	CollisionSystem::CollisionSystem()
		: m_staticRootNode(NULL),
		  m_dynamicRootNode(NULL)
	{

	}

	CollisionSystem::~CollisionSystem()
	{

	}

	void CollisionSystem::AddNewListeners()
	{
		for(auto it = m_newListeners.begin(); it != m_newListeners.end();)
		{
			if((*it)->m_isInit)
			{
				if((*it)->GetType() == CL_STATIC)
				{
					AddStaticListener(*it);
					m_staticListeners.push_back(*it);
				}
				else
				{
					AddDynamicListener(*it);
					m_dynamicListeners.push_back(*it);
				}

				it = m_newListeners.erase(it);
			}
			else
				++it;
		}
	}

	void CollisionSystem::AddListenerToTree(CollisionListener* listener, CollisionNode* currentNode)
	{
		if(currentNode->m_listener != NULL)
		{
			CollisionNode* newNodeLeft = new CollisionNode();
			newNodeLeft->m_listener = listener;
			newNodeLeft->m_cost = listener->m_boundingVolume->SurfaceArea();

			CollisionNode* newNodeRight = new CollisionNode();
			newNodeRight->m_listener = currentNode->m_listener;
			newNodeRight->m_cost = currentNode->m_listener->m_boundingVolume->SurfaceArea();

			currentNode->m_left = newNodeLeft;
			currentNode->m_right = newNodeRight;
			currentNode->m_aabb.CombineAABB(newNodeLeft->GetAABB(), newNodeRight->GetAABB());
			currentNode->m_listener = NULL;
		}
		else
		{
			AABoundingBox newAABB = listener->m_boundingVolume->ConvertToAABB();
			AABoundingBox bb;
			Real leftCost;
			Real rightCost;

			bb.CombineAABB(newAABB, currentNode->m_left->GetAABB());
			leftCost = bb.SurfaceArea();

			bb.CombineAABB(newAABB, currentNode->m_right->GetAABB());
			rightCost = bb.SurfaceArea();

			if(leftCost < rightCost)
				AddListenerToTree(listener, currentNode->m_left);
			else
				AddListenerToTree(listener, currentNode->m_right);
		}

		currentNode->m_aabb.CombineAABB(currentNode->m_left->GetAABB(), currentNode->m_right->GetAABB());
		currentNode->m_cost = currentNode->m_aabb.SurfaceArea();
	}

	void CollisionSystem::AddStaticListener(CollisionListener* listener)
	{
		if(!m_staticRootNode)
		{
			m_staticRootNode = new CollisionNode();
			m_staticRootNode->m_listener = listener;
			m_staticRootNode->m_cost = listener->GetBV()->SurfaceArea();
		}
		else
		{
			AddListenerToTree(listener, m_staticRootNode);
		}
	}

	void CollisionSystem::AddDynamicListener(CollisionListener* listener)
	{
		if(!m_dynamicRootNode)
		{
			m_dynamicRootNode = new CollisionNode();
			m_dynamicRootNode->m_listener = listener;
			m_dynamicRootNode->m_cost = listener->GetBV()->SurfaceArea();
		}
		else
		{
			AddListenerToTree(listener, m_dynamicRootNode);
		}
	}

	static int numCollisionChecks = 0;
	void CollisionSystem::DetectCollisionsP(CollisionListener* listener, CollisionNode* currentNode)
	{
		if(!currentNode || listener == currentNode->m_listener)
			return;

		BoundingVolume* currentNodeBV;

		if(currentNode->m_listener)
			currentNodeBV = currentNode->m_listener->GetBV();
		else
			currentNodeBV = &currentNode->m_aabb;

		if(BoundingVolumeCollision(listener->GetBV(), currentNodeBV))
		{
			++numCollisionChecks;
			if(currentNode->m_listener)
			{
				AddCollision(listener, currentNode->m_listener);
			}
			else
			{
				DetectCollisionsP(listener, currentNode->m_left);
				DetectCollisionsP(listener, currentNode->m_right);
			}
		}
	}

	void CollisionSystem::DetectDynamicToStaticCollisions()
	{
		numCollisionChecks = 0;

		for(size_t i = 0; i < m_dynamicListeners.size(); ++i)
		{
			DetectCollisionsP(m_dynamicListeners[i], m_staticRootNode);
		}

		//trace("Actual Num Collision Checks: " << numCollisionChecks)
		//trace("Worst Case Num Collision Checks: " << m_staticListeners.size() << "\n")
	}

	void CollisionSystem::DetectDynamicToDynamicCollisions()
	{
		numCollisionChecks = 0;

		for(size_t i = 0; i < m_dynamicListeners.size(); ++i)
		{
			DetectCollisionsP(m_dynamicListeners[i], m_dynamicRootNode);
		}

	//	trace("Actual Num Collision Checks: " << numCollisionChecks)
	//	trace("Worst Case Num Collision Checks: " << m_dynamicListeners.size() * m_dynamicListeners.size() << "\n")
	}

	void CollisionSystem::AddCollision(CollisionListener* a, CollisionListener* b)
	{
		REQUIRES(a)
		REQUIRES(b)

		AlertListeners(a, b);

		if((a->m_isActive || b->m_isActive))
			m_collisions.push_back(std::pair<CollisionListener*, CollisionListener*>(a, b));
	}

	void CollisionSystem::AlertListeners(CollisionListener* a, CollisionListener* b)
	{
		a->Collided(b);
		b->Collided(a);
	}

	//////////////////////////////////////////////////////////////////////////
	//	Private
	//////////////////////////////////////////////////////////////////////////
}
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

#include "CollisionListener.h"

#include "CollisionSystem.h"
#include "BoundingVolumes/BoundingVolume.h"

#include "Utils/UtilDebug.h"

namespace CaptainLucha
{
	CollisionListener::CollisionListener()
		: m_isInit(false),
		  m_isActive(true),
		  m_primitive(NULL),
		  m_boundingVolume(NULL),
		  m_type(CL_TRIGGER),
		  m_collisionSystem(NULL)
	{

	}

	CollisionListener::~CollisionListener()
	{
		m_collisionSystem->RemoveListener(this);
	}

	void CollisionListener::InitListener(BoundingVolume* aabb, Primitive* primitive)
	{
		REQUIRES(aabb)
		REQUIRES(primitive)

		m_primitive = primitive;
		m_boundingVolume = aabb;
		m_isInit = true;
	}
}
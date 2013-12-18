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
 *	@file	CollisionListener.h
 *  @todo	Switch to component design
 *	@brief	
 *
/****************************************************************************/

#ifndef COLLISIONLISTENER_H_CL
#define COLLISIONLISTENER_H_CL

#include "Utils/UtilMacros.h"
#include "Objects/Object.h"
#include "BoundingVolumes/BoundingVolume.h"
#include "Physics/Primitives/Primitive.h"
#include "Utils/CommonIncludes.h"
#include "Collision/CollisionFunctions.h"

namespace CaptainLucha
{		
   /**
	* @enum	
	* @brief	Holds information about a CollisionLisener's type.     
	*/
	enum CollisionListenerType
	{
		CL_TRIGGER,
		CL_STATIC,
		CL_MOVEABLE,
		CL_PHYSICS_OBJECT
	};

	class CollisionSystem;

   /**
	* @brief Allows for collision detection for the object. No resolution for any collisions.
	* @see CollisionSystem CollisionListener::Collided
	* @attention Must call InitListener
	*/
	class CollisionListener : public Object
	{
	public:
		CollisionListener();
		~CollisionListener();

		/**
		 * @brief     Returns the BoundingVolume associated with this CollisionListener
		 * @return    const BoundingVolume*
		 */
		const BoundingVolume* GetBV() const {return m_boundingVolume;}
		BoundingVolume* GetBV() {return m_boundingVolume;}

		/**
		 * @brief     Returns the Primitive associated with this CollisionListener
		 * @return    const Primitive*
		 */
		const Primitive* GetPrimitve() const {return m_primitive;}
		Primitive* GetPrimitve() {return m_primitive;}

		/**
		 * @brief     Initializes the listener
		 * @param	  BoundingVolume * aabb
		 * @param	  Primitive * primitive
		 * @todo	  Better doc
		 */
		void InitListener(BoundingVolume* aabb, Primitive* primitive);

		/**
		 * @brief     Returns if the CollisionListener's BoundingVolume is colliding with another CollisionListener. This is usually called from CollisionSystem.
		 * @param	  CollisionListener * otherListener
		 * @see		  CollisionSystem
		 * @pure
		 */
		virtual void Collided(CollisionListener* otherListener) {UNUSED(otherListener)}
		
		/**
		 * @brief     If a CollisionListener is not active, it will not create collisions.
		 */
		bool IsActive() const {return m_isActive;}
		void SetIsActive(bool val) {m_isActive = val;}

		/**
		 * @brief     Returns the listener's type. Subclasses of CollisionLisener should set the type.
		 * @return    CaptainLucha::CollisionListenerType
		 */
		CollisionListenerType GetType() const {return m_type;}

	protected:
		CollisionListenerType m_type;

	private:
		CollisionSystem* m_collisionSystem;
		BoundingVolume* m_boundingVolume;
		Primitive* m_primitive;

		bool m_isInit;
		bool m_isActive;

		friend class CollisionSystem;
	};
}

#endif
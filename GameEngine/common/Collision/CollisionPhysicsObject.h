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
 *	@file	CollisionPhysicsObject.h
 *  @todo	Let initialize functions override any other previous initialize?
 *  @see	RigidBody Primitive BoundingVolume
 *	@brief	
 *
/****************************************************************************/

#ifndef COLLISION_PHYSICSOBJECT_H_CL
#define COLLISION_PHYSICSOBJECT_H_CL

#include "Physics/RigidBody.h"
#include "Physics/Primitives/Primitive.h"
#include "Collision/BoundingVolumes/AABoundingBox.h"
#include "Collision/CollisionListener.h"
#include "Renderer/Geometry/Sphere.h"

namespace CaptainLucha
{
	class CollisionSystem;

   /**
	* @brief Object that simulates rigid body physics and reacts to collisions
	* @see CollisionSystem PhysicsSystem
	*/
	class CollisionPhysicsObject : public CollisionListener
	{
	public:
		CollisionPhysicsObject();
		~CollisionPhysicsObject();

		void Update(Real DT);

		/**
		 * @brief     Draws debug solid sphere or cube. Should be overridden to draw actual mesh.
		 * @return    void
		 */
		virtual void Draw(GLProgram& glProgram);

		/**
		 * @brief     Initializes this object to a cube.
		 * @attention Initialize only once.
		 * @return    void
		 */
		void InitCuboid(const Vector3D<Real>& extent, Real mass);
		/**
		 * @brief     Initializes this object to a sphere.
		 * @attention Initialize only once.
		 * @return    void
		 */
		void InitSphere(Real radius, Real mass);

		/**
		 * @brief     Called when this object has collided with another object.
		 * @return    void
		 */
		virtual void Collided(CollisionListener* listener);

		void SetPosition(const Vector3D<Real>& pos);
		const Vector3D<Real>& GetPosition() const {return m_body.GetPosition();}

		const Quaternion& GetOrientation() const {return m_body.GetOrientation();}

		const Matrix4D<Real>& GetTransformation() const {return m_body.GetTransformation();}

		const RigidBody& GetRigidBody() const {return m_body;}
		RigidBody& GetRigidBody() {return m_body;}

		bool IsInit() const {return m_isInit;}

	protected:

	private:
		RigidBody m_body;
		BoundingVolume* m_boundingVolume;
		Primitive* m_primitive;

		Sphere m_drawSphere;

		bool m_isInit;
	};
}

#endif
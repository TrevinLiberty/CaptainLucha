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
 *	@file	RigidBody.h
 *	@brief	
 *
/****************************************************************************/

#ifndef RIGIDBODY_H_CL
#define RIGIDBODY_H_CL

#include <Math/Quaternion.h>
#include <Objects/Object.h>
#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class RigidBody : public Object
	{
	public:
		RigidBody();
		~RigidBody();

		/**
		 * @brief     Integrates the RigidBody forward by DT
		 */
		void Integrate(double DT);

		/**
		 * @brief     Applies force to the center of the RigidBody
		 * @param	  const Vector3D<Real>& force The Direction and Magnitude of the force
		 */
		void ApplyForce(const Vector3D<Real>& force);

		/**
		* @brief      Applies force at a worldPos of the RigidBody
		* @param	  const Vector3D<Real>& force The Direction and Magnitude of the force
		 */
		void ApplyForceToWorldPoint(const Vector3D<Real>& worldPos, const Vector3D<Real>& force);

		/**
		 * @brief     Applies force at a localPos of the RigidBody
		 * @param	  const Vector3D<Real>& force The Direction and Magnitude of the force
		 */
		void ApplyForceToLocalPoint(const Vector3D<Real>& localPos, const Vector3D<Real>& force);

		/**
		 * @brief     Sets the RigidBody as Asleep or Awake. RigidBodies that are asleep do not update.
		 */
		void SetAwakeState(bool val);
		bool IsAwake() const {return m_isAwake;}

		/**
		 * @brief     Wakes up the RigidBody.
		 */
		void Wake();


		/**
		 * @brief     Sets the inertia tensor matrix.
		 */
		void SetInverseInertiaTensor(const Matrix3D<Real>& val) {m_invInertiaTensor = val;}
		void SetInertiaTensor(const Matrix3D<Real>& val) { m_invInertiaTensor = val.GetInverse();}

		/**
		 * @brief     Initializes the inertia tensor as a cuboid
		 */
		void SetInertiaTensorCuboid(Real width, Real height, Real depth, Real mass);

		/**
		 * @brief     Initializes the inertia tensor as a sphere
		 */
		void SetInertiaTensorSphere(Real radius, Real mass);

		/**
		 * @brief     Sets the Rigidbody's current linear acceleration
		 */
		void SetAcceleration(const Vector3D<Real>& accell) {m_acceleration = accell;}

		const Matrix3D<Real>& GetInverseInertiaTensor() const {return m_invInertiaTensor;}
		const Matrix3D<Real>& GetInverseInertiaTensorWorld() const {return m_invInertiaTensorWorld;}

		const Matrix4D<Real>& GetTransformation() const {return m_transformMatrix;}

		/**
		 * @brief     Calculates the RigidBody's current transformation matrix using position and orientation
		 */
		void CalculateTransformation();

		/**
		 * @brief     Calculates the RigidBody's current transformation and InverseInertiaTensorWorld matrix
		 */
		void CalculatedDerivedData();

		void SetLinearDamp(Real val) {m_linearDamp = val;}
		void SetAngularDamp(Real val) {m_angularDamp = val;}

		void SetOrientation(const Quaternion& val) {m_orientation = val;}
		void AddToOrientationWithVector(const Vector3D<Real>& val) {m_orientation.AddScaledVector(val, 1.0f);}
		const Quaternion& GetOrientation() const {return m_orientation;}

		Real GetMass() const {return m_mass;}
		Real GetInvMass() const {return m_invMass;}

		const Vector3D<Real>& GetAngularVelocity() const {return m_angularVelocity;}
		const Vector3D<Real>& GetVelocity() const {return m_velocity;}

		void SetAngularVelocity(const Vector3D<Real>& velocity) {m_angularVelocity = velocity;}
		void SetVelocity(const Vector3D<Real>& velocity) {m_velocity = velocity;}

		void AddAngularVelocity(const Vector3D<Real>& velocity) {m_angularVelocity += velocity;}
		void AddVelocity(const Vector3D<Real>& velocity) {m_velocity += velocity;}

		/**
		 * @brief     Returns the Acceleration as of last frame
		 */
		const Vector3D<Real>& GetLastFrameAcceleration() const {return m_lastFrameAcceleration;}

		Vector3D<Real> GetWorldSpacePosition(const Vector3D<Real>& pos);

	protected:
		void CalculateInvInteriaTensorWorld();

		void ClearAccumulators();

		void UpdateSleepMode();

	private:
		bool m_isAwake;

		Real m_mass;
		Real m_invMass;
		Real m_linearDamp;
		Real m_angularDamp;

		Real m_avgLinearVelocity;
		Real m_avgAngularVelocity;

		Vector3D<Real> m_velocity;
		Vector3D<Real> m_acceleration;
		Vector3D<Real> m_lastFrameAcceleration;
		Vector3D<Real> m_angularVelocity;

		Vector3D<Real> m_forceAccunulator;
		Vector3D<Real> m_torqueAccumulator;

		Quaternion m_orientation;

		//for body-><-world space transforms.
		Matrix4D<Real> m_transformMatrix;
		Matrix3D<Real> m_invInertiaTensor;
		Matrix3D<Real> m_invInertiaTensorWorld;

		static Real SLEEP_SQRD_LINEAR_VELOCITY;
		static Real SLEEP_SQRD_ANGULAR_VELOCITY;
	};
}

#endif
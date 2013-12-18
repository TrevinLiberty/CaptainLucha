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

#ifndef CONTACT_H_CL
#define CONTACT_H_CL

#include "Math/Vector3D.h"
#include "Math/Matrix4D.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class RigidBody;

	class Contact
	{
	public:
		Contact();
		~Contact();

		RigidBody* m_body1;
		RigidBody* m_body2;

		Real m_penetration;

		Vector3D<Real> m_contactPoint;
		Vector3D<Real> m_contactNormal;

		//Quick Hack. Fix
		bool m_switchBodies;

		void InitContact(Real DT);

		void ResolveVelocity(Vector3D<Real>* linearVelocityChange, Vector3D<Real>* angularVelocityChange);

		void ResolvePositions(Vector3D<Real>* linearChange, Vector3D<Real>* angularChange, Real penetration);

		void MatchAwakeState();

	protected:
		void CalculateContactBasis();

		//////////////////////////////////////////////////////////////////////////
		//	Velocity Change Functions
		Vector3D<Real> CalculateContactImpulseWorld();
		Vector3D<Real> CalculateContactImpulseWorldFriction();

		void CalculateDesiredDeltaVelocity(Real duration);

		void CalculateContactVelocity(Real duration);

		Vector3D<Real> CalculateContactVelocity(RigidBody* body, const Vector3D<Real>& relativePos);
		
		Real CalculateDeltaVelPerUnitImpulse(RigidBody* body, const Vector3D<Real>& relativePos);
		Matrix3D<Real> CalculateDeltaVelPerUnitImpulseFriction(RigidBody* body, const Vector3D<Real>& relativePos);

		void ApplyVelocityChange(const Vector3D<Real>& impulseWorld, Vector3D<Real>* linearVelocityChange, Vector3D<Real>* angularVelocityChange);

		//////////////////////////////////////////////////////////////////////////
		//	Position Change Functions
		void CalculateLinearAngularMove(Vector3D<Real>* linearChange, Vector3D<Real>* angularChange, Real penetration);

		Real CalculateInertia(RigidBody* body, const Vector3D<Real>& relativePos, Real& linearInertia, Real& angularInertia);

		void AvoidOverRotation(Real& angularMove, Real& linearMove, const Vector3D<Real>& relativePos);

		Vector3D<Real> CalculateAddRotation(RigidBody* body, const Vector3D<Real>& relativePos, Real angularMove, Real angularInertia);

		Matrix3D<Real> m_contactToWorldTrans;

		Vector3D<Real> m_contactVelocity;
		Real m_desiredDeltaVelocity;

		Vector3D<Real> m_relativePos1;
		Vector3D<Real> m_relativePos2;

		friend class ContactResolver;

	};
}

#endif
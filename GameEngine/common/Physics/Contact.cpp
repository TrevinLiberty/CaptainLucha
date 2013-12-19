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
 *  @attention Significant resource for implementation: http://procyclone.com/
 *
/****************************************************************************/

#include "Contact.h"

#include "Math.h"
#include "RigidBody.h"

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	Contact::Contact()
		: m_switchBodies(false),
		  m_body1(NULL),
		  m_body2(NULL)
	{

	}

	Contact::~Contact()
	{

	}

	void Contact::InitContact(Real DT)
	{
		CalculateContactBasis();
		CalculateContactVelocity(DT);
		CalculateDesiredDeltaVelocity(DT);
	}

	void Contact::ResolveVelocity(Vector3D<Real>* linearVelocityChange, Vector3D<Real>* angularVelocityChange)
	{
		ApplyVelocityChange(CalculateContactImpulseWorldFriction(), linearVelocityChange, angularVelocityChange);
		//ApplyVelocityChange(CalculateContactImpulseWorld(), linearVelocityChange, angularVelocityChange);
	}

	void Contact::ResolvePositions(Vector3D<Real>* linearChange, Vector3D<Real>* angularChange, Real penetration)
	{
		CalculateLinearAngularMove(linearChange, angularChange, penetration);
	}

	void Contact::MatchAwakeState()
	{
		if(m_body1)
		{
			if(m_body1->IsAwake() && m_body2)
				m_body2->SetAwakeState(true);
		}

		if(m_body2)
		{
			if(m_body2->IsAwake() && m_body1)
				m_body1->SetAwakeState(true);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void Contact::CalculateContactBasis()
	{
		m_contactNormal.Normalize();

		Vector3D<Real> x(m_contactNormal);
		Vector3D<Real> y(0.0f, 1.0f, 0.0f);
		Vector3D<Real> z;

		if(std::abs(x.x) < std::abs(x.y))
		{
			y = Vector3D<Real>(1.0f, 0.0f, 0.0f);
			MakeOrthonormalBasis(x, y, z);
			m_contactToWorldTrans = Matrix3D<Real>(x, y, z);
		}
		else
		{
			MakeOrthonormalBasis(x, y, z);
			m_contactToWorldTrans = Matrix3D<Real>(x, y, z * -1);
		}
	}

	Vector3D<Real> Contact::CalculateContactImpulseWorld()
	{
		Real deltaVelocity = 0.0f;

		if(m_body1)
			deltaVelocity += CalculateDeltaVelPerUnitImpulse(m_body1, m_relativePos1);

		if(m_body2)
			deltaVelocity += CalculateDeltaVelPerUnitImpulse(m_body2, m_relativePos2);

		return m_contactToWorldTrans * Vector3D<Real>(m_desiredDeltaVelocity / deltaVelocity, 0.0f, 0.0f);
	}

	Vector3D<Real> Contact::CalculateContactImpulseWorldFriction()
	{
		const Real FRICTION = 10.0f;
		Matrix3D<Real> deltaVelocityWorld;
		Real inverseMass = 0.0f;

		if(m_body1)
		{
			deltaVelocityWorld = CalculateDeltaVelPerUnitImpulseFriction(m_body1, m_relativePos1);
			inverseMass = m_body1->GetInvMass();
		}

		if(m_body2)
		{
			deltaVelocityWorld += CalculateDeltaVelPerUnitImpulseFriction(m_body2, m_relativePos2);
			inverseMass += m_body2->GetInvMass();
		}

		Matrix3D<Real> deltaVelocity = m_contactToWorldTrans.GetTranspose();
		deltaVelocity = deltaVelocity * deltaVelocityWorld;
		deltaVelocity = deltaVelocity * m_contactToWorldTrans;
		
		deltaVelocity[0] += inverseMass;
		deltaVelocity[4] += inverseMass;
		deltaVelocity[8] += inverseMass;

		Matrix3D<Real> impulseMatrix = deltaVelocity.GetInverse().GetTranspose();

		Vector3D<Real> velKill(m_desiredDeltaVelocity, -m_contactVelocity.y, -m_contactVelocity.z);

		Vector3D<Real> impulseContact = impulseMatrix * velKill;

		Real planarImpulse = std::sqrt(impulseContact.y * impulseContact.y + impulseContact.z * impulseContact.z);

		if(planarImpulse > impulseContact.x * FRICTION)
		{
			impulseContact.y /= planarImpulse;
			impulseContact.z /= planarImpulse;
			impulseContact.x = deltaVelocity[0] +
			deltaVelocity[1]*FRICTION*impulseContact.y +
			deltaVelocity[2]*FRICTION*impulseContact.z;
			impulseContact.x = m_desiredDeltaVelocity / impulseContact.x;
			impulseContact.y *= FRICTION * impulseContact.x;
			impulseContact.z *= FRICTION * impulseContact.x;
		}

		return m_contactToWorldTrans * impulseContact;
	}

	void Contact::CalculateContactVelocity(Real duration)
	{
		m_contactVelocity = Vector3D<Real>();

		Matrix3D<Real> worldToContact = m_contactToWorldTrans.GetTranspose();

		if(m_body1)
		{
			Vector3D<Real> velocity = CalculateContactVelocity(m_body1, m_relativePos1);

			Vector3D<Real> worldLastAcc(m_body1->GetLastFrameAcceleration() * duration);
			Vector3D<Real> constactLastAcc = worldToContact * worldLastAcc;
			constactLastAcc.x = 0.0f;

			m_contactVelocity += (velocity + constactLastAcc);

		}

		if(m_body2)
		{
			Vector3D<Real> velocity = CalculateContactVelocity(m_body2, m_relativePos2);

			Vector3D<Real> worldLastAcc(m_body2->GetLastFrameAcceleration() * duration);
			Vector3D<Real> constactLastAcc = worldToContact * worldLastAcc;
			constactLastAcc.x = 0.0f;

			m_contactVelocity -= (velocity + constactLastAcc);
		}
	}

	Vector3D<Real> Contact::CalculateContactVelocity(RigidBody* body, const Vector3D<Real>& relativePos)
	{
		Vector3D<Real> velocity = body->GetAngularVelocity().CrossProduct(relativePos);
 		Matrix3D<Real> m_worldToContactTrans = m_contactToWorldTrans.GetTranspose();
		return m_worldToContactTrans * (velocity + body->GetVelocity());
	}

	void Contact::CalculateDesiredDeltaVelocity(Real duration)
	{
		Real restitution = 0.4f;
		if(abs(m_contactVelocity.x) < 0.1f)
			restitution = 0.0f;

		Real velFromAcc = 0.0f;

		if(m_body1 && m_body1->IsAwake())
		{
			velFromAcc += (m_body1->GetLastFrameAcceleration() * duration).Dot(m_contactNormal);
		}

		if(m_body2 && m_body2->IsAwake())
		{
			velFromAcc -= (m_body2->GetLastFrameAcceleration() * duration).Dot(m_contactNormal);
		}

		m_desiredDeltaVelocity = -m_contactVelocity.x - restitution * (m_contactVelocity.x - velFromAcc);
	}

	Matrix3D<Real> Contact::CalculateDeltaVelPerUnitImpulseFriction(RigidBody* body, const Vector3D<Real>& relativePos)
	{
		UNUSED(body)
		Matrix3D<Real> impulseToTorque(true);
		impulseToTorque.SetSkewSymmetric(relativePos);

		Matrix3D<Real> deltaVelocityWorld = impulseToTorque;
		deltaVelocityWorld = m_body1->GetInverseInertiaTensorWorld() * deltaVelocityWorld;
		deltaVelocityWorld = impulseToTorque * deltaVelocityWorld;
		deltaVelocityWorld *= -1;

		return deltaVelocityWorld;
	}

	Real Contact::CalculateDeltaVelPerUnitImpulse(RigidBody* body, const Vector3D<Real>& relativePos)
	{
		Vector3D<Real> deltaVelWorld = relativePos.CrossProduct(m_contactNormal);
		Vector3D<Real> rotationPerUnitImpulse = body->GetInverseInertiaTensorWorld() * deltaVelWorld;
		Vector3D<Real> velocityPerUnitImpulse = rotationPerUnitImpulse.CrossProduct(relativePos);
		Vector3D<Real> velocityPerUnitImpulseContact = m_contactToWorldTrans.GetTranspose() * velocityPerUnitImpulse;

		Real deltaVelocity = velocityPerUnitImpulse.Dot(m_contactNormal);
		deltaVelocity += body->GetInvMass();
		return deltaVelocity;
	}

	void Contact::ApplyVelocityChange(const Vector3D<Real>& impulse, Vector3D<Real>* linearVelocityChange, Vector3D<Real>* angularVelocityChange)
	{
		if(m_body1)
		{
			const Vector3D<Real> impulsiveTorque = m_relativePos1.CrossProduct(impulse);

			angularVelocityChange[0] = m_body1->GetInverseInertiaTensorWorld() * impulsiveTorque;
			linearVelocityChange[0] = impulse * m_body1->GetInvMass();

			m_body1->AddAngularVelocity(angularVelocityChange[0]);
			m_body1->AddVelocity(linearVelocityChange[0]);
		}

		if(m_body2)
		{
			const Vector3D<Real> impulsiveTorque = impulse.CrossProduct(m_relativePos2);

			angularVelocityChange[1] = m_body2->GetInverseInertiaTensorWorld() * impulsiveTorque;
			linearVelocityChange[1] = impulse * -m_body2->GetInvMass();

			m_body2->AddAngularVelocity(angularVelocityChange[1]);
			m_body2->AddVelocity(linearVelocityChange[1]);
		}
	}

	void Contact::CalculateLinearAngularMove(Vector3D<Real>* linearChange, Vector3D<Real>* angularChange, Real penetration)
	{
		//////////////////////////////////////////////////////////////////////////
		//	Calculate Inertia
		Real linearInertia[2];
		Real angularInertia[2];

		Real totalInertia = 0.0f;

		if(m_body1)
			totalInertia += CalculateInertia(m_body1, m_relativePos1, linearInertia[0], angularInertia[0]);
		
		if(m_body2)
			totalInertia += CalculateInertia(m_body2, m_relativePos2, linearInertia[1], angularInertia[1]);

		Real inverseInertia = 1.0f / totalInertia;

		//////////////////////////////////////////////////////////////////////////
		//	Calculate Move
		Real linearMove[2];
		Real angularMove[2];

		linearMove[0] =  penetration * linearInertia[0] * inverseInertia;
		linearMove[1] = -penetration * linearInertia[1] * inverseInertia;
		angularMove[0] =  penetration * angularInertia[0] * inverseInertia;
		angularMove[1] = -penetration * angularInertia[1] * inverseInertia;

		AvoidOverRotation(angularMove[0], linearMove[0], m_relativePos1);
		AvoidOverRotation(angularMove[1], linearMove[1], m_relativePos2);

		//////////////////////////////////////////////////////////////////////////
		//	Add Linear Move
		if(m_body1)
		{
			linearChange[0] = m_contactNormal * linearMove[0];
			m_body1->AddToPosition(linearChange[0]);
		}

		if(m_body2)
		{
			linearChange[1] = m_contactNormal * linearMove[1];
			m_body2->AddToPosition(linearChange[1]);
		}

		//////////////////////////////////////////////////////////////////////////
		//Add Angular Move
		if(m_body1)
		{
			if(FloatEquality(angularInertia[0], 0.0f))
				angularChange[0] = Vector3D<Real>();
			else
			{
				angularChange[0] = CalculateAddRotation(m_body1, m_relativePos1, angularMove[0], angularInertia[0]);
				m_body1->AddToOrientationWithVector(angularChange[0]);
			}
		}

		if(m_body2)
		{
			if(FloatEquality(angularInertia[1], 0.0f))
				angularChange[1] = Vector3D<Real>();
			else
			{
				angularChange[1] = CalculateAddRotation(m_body2, m_relativePos2, angularMove[1], angularInertia[1]);
				m_body2->AddToOrientationWithVector(angularChange[1]);
			}
		}

		if(m_body1 && !m_body1->IsAwake())
			m_body1->CalculatedDerivedData();
		if(m_body2 && !m_body2->IsAwake())
			m_body2->CalculatedDerivedData();
	}

	Real Contact::CalculateInertia(RigidBody* body, const Vector3D<Real>& relativePos, Real& linearInertia, Real& angularInertia)
	{
		const Matrix3D<Real>& inverseInertiaTensor = body->GetInverseInertiaTensorWorld();

		Vector3D<Real> angularInertiaWorld = relativePos.CrossProduct(m_contactNormal);
		angularInertiaWorld = inverseInertiaTensor * angularInertiaWorld;
		angularInertiaWorld = angularInertiaWorld.CrossProduct(relativePos);

		angularInertia = angularInertiaWorld.Dot(m_contactNormal);
		linearInertia = body->GetInvMass();

		return angularInertia + linearInertia;
	}

	void Contact::AvoidOverRotation(Real& angularMove, Real& linearMove, const Vector3D<Real>& relativePos)
	{
		Real ANGULAR_LIMIT = 0.01;
		Vector3D<Real> projection = relativePos;
		projection.CrossProduct(m_contactNormal * -relativePos.Dot(m_contactNormal));

		Real maxMagnitude = ANGULAR_LIMIT * projection.Length();

		if(angularMove < -maxMagnitude)
		{
			Real totalMove = angularMove + linearMove;
			angularMove = -maxMagnitude;
			linearMove = totalMove - angularMove;
		}
		else if(angularMove > maxMagnitude)
		{
			Real totalMove = angularMove + linearMove;
			angularMove = maxMagnitude;
			linearMove = totalMove - angularMove;
		}
	}

	Vector3D<Real> Contact::CalculateAddRotation(RigidBody* body, const Vector3D<Real>& relativePos, Real angularMove, Real angularInertia)
	{
		const Matrix3D<Real>& inverseInertiaTensor = body->GetInverseInertiaTensorWorld();
		Vector3D<Real> targetAngularDirection = relativePos.CrossProduct(m_contactNormal);
		targetAngularDirection = inverseInertiaTensor * targetAngularDirection;
		return targetAngularDirection * (angularMove / angularInertia);
	}
}
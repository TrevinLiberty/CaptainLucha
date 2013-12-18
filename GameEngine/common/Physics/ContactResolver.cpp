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

#include "ContactResolver.h"

#include "Collision/CollisionFunctions.h"

namespace CaptainLucha
{
	ContactResolver::ContactResolver(RigidBody* body1, Primitive* primitive1, RigidBody* body2, Primitive* primitive2)
		: m_body1(body1),
		  m_body2(body2),
		  m_primitive1(primitive1),
		  m_primitive2(primitive2)
	{
	}

	ContactResolver::~ContactResolver()
	{

	}

	void ContactResolver::GenerateContacts()
	{	
		bool a = true;
		bool b = true;

		if(m_body1)
			a = m_body1->IsAwake();

		if(m_body2)
			b = m_body2->IsAwake();

		if(!a && !b)
			return;

		GeneratePrimitiveContacts(m_primitive1, m_primitive2, m_contacts);
		for(size_t i = 0; i < m_contacts.size(); ++i)
		{
			m_contacts[i].m_body1 = m_body1;
			m_contacts[i].m_body2 = m_body2;

			if(m_body1)
				m_contacts[i].m_relativePos1 = m_contacts[i].m_contactPoint - m_contacts[i].m_body1->GetPosition();

			if(m_body2)
				m_contacts[i].m_relativePos2 = m_contacts[i].m_contactPoint - m_contacts[i].m_body2->GetPosition();
		}
	}

	void ContactResolver::ResolveContacts()
	{
		if(!m_contacts.empty())
		{
			InitContacts();
			ResolvePosition();
			ResolveVelocity();
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void ContactResolver::InitContacts()
	{
		for(size_t i = 0; i < m_contacts.size(); ++i)
		{
			m_contacts[i].InitContact(0.01666);
		}
	}

	void ContactResolver::ResolvePosition()
	{
		const int MAX_ITERATIONS = 100;
		const Real POSITION_EPSILON = 0.1f;
		
		int index;
		int positionIterationsUsed = 0;
		Vector3D<Real> linearChange[2];
		Vector3D<Real> angularChange[2];
		Real max;
		Vector3D<Real> deltaPosition;

		while(positionIterationsUsed < MAX_ITERATIONS)
		{
			max = POSITION_EPSILON;
			index = (int)m_contacts.size();

			for(size_t i = 0; i < m_contacts.size(); ++i)
			{
				if(m_contacts[i].m_penetration > max)
				{
					max = m_contacts[i].m_penetration;
					index = i;
				}
			}

			if(index == (int)m_contacts.size())
				break;

			m_contacts[index].MatchAwakeState();
			m_contacts[index].ResolvePositions(linearChange, angularChange, max);

			for(size_t i = 0; i < m_contacts.size(); ++i)
			{
				if(m_body1)
					UpdateBodyPenetration(m_body1, m_contacts[i], -1, m_contacts[i].m_relativePos1, linearChange[0], angularChange[0]);

				if(m_body2)
					UpdateBodyPenetration(m_body2, m_contacts[i], 1, m_contacts[i].m_relativePos2, linearChange[1], angularChange[1]);
			}

			++positionIterationsUsed;
		}
	}

	void ContactResolver::ResolveVelocity()
	{
		const int MAX_VEL_ITERATIONS = 100;
		const Real VEL_EPSILON = 0.1f;

		Vector3D<Real> linearVelocityChange[2], angularVelocityChange[2];
		Vector3D<Real> deltaVel;
		int velocityIterationsUsed = 0;

		while(velocityIterationsUsed < MAX_VEL_ITERATIONS)
		{
			Real max = VEL_EPSILON;
			unsigned int index = m_contacts.size();
			for(size_t i = 0; i < m_contacts.size(); ++i)
			{
				if(m_contacts[i].m_desiredDeltaVelocity > max)
				{
					max = m_contacts[i].m_desiredDeltaVelocity;
					index = i;
				}
			}

			if(index == m_contacts.size())
				break;

			m_contacts[index].MatchAwakeState();
			m_contacts[index].ResolveVelocity(linearVelocityChange, angularVelocityChange);
		
			for(size_t i = 0; i < m_contacts.size(); ++i)
			{
				Matrix3D<Real> worldToContact = m_contacts[i].m_contactToWorldTrans.GetTranspose();

				if(m_body1)
				{
					deltaVel = linearVelocityChange[0] + angularVelocityChange[0].CrossProduct(m_contacts[i].m_relativePos1);
					m_contacts[i].m_contactVelocity += worldToContact * deltaVel;
					m_contacts[i].CalculateDesiredDeltaVelocity(0.01666);
				}

				if(m_body2)
				{
					deltaVel = linearVelocityChange[1] + angularVelocityChange[1].CrossProduct(m_contacts[i].m_relativePos2);
					m_contacts[i].m_contactVelocity -= worldToContact * deltaVel;
					m_contacts[i].CalculateDesiredDeltaVelocity(0.01666);
				}
			}

			++velocityIterationsUsed;
		}
	}

	void ContactResolver::UpdateBodyPenetration(
		RigidBody* body, Contact& contact, int sign, 
		const Vector3D<Real>& relativePos, const Vector3D<Real>& linearChange, 
		const Vector3D<Real>& angularChange)
	{
		UNUSED(body)
		Vector3D<Real> deltaPosition = linearChange + angularChange.CrossProduct(relativePos);
		contact.m_penetration += deltaPosition.Dot(contact.m_contactNormal) * sign;
	}
}
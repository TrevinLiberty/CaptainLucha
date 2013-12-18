///////////////////////////////////////////////////////////////////////
//
//	Author: Trevin Liberty
//	Data:   9/4/2013
//	Description:
//		
//
///////////////////////////////////////////////////////////////////////

#ifndef CABLE_H_CL
#define CABLE_H_CL

#include "ContactGenerator.h"

namespace CaptainLucha
{
	class Cable : public ContactGenerator
	{
	public:
		Cable(RigidBody* body, const Vector3D<Real>& localBodyPoint)
		  : m_body(body),
		  m_localBodyPos(localBodyPoint) {REQUIRES(body)};
		~Cable() {};

		inline void SetCableLength(Real length) {m_cableLength = length;}
		inline void SetRestitution(Real rest) {m_restitution = rest;}

	protected:
		Real m_cableLength;
		Real m_restitution;

		RigidBody* m_body;
		Vector3D<Real> m_localBodyPos;

	private:

	};

	class CableRigidBodyToPoint : public Cable
	{
	public:
		CableRigidBodyToPoint(
			RigidBody* body, const Vector3D<Real>& localBodyPoint, 
			const Vector3D<Real>& worldConnectionPoint)
			: Cable(body, localBodyPoint),
			  m_worldConnectionPoint(worldConnectionPoint) {}
		~CableRigidBodyToPoint() {};

		virtual bool GetNewContact(Contact& newContact) const
		{
			const Vector3D<Real> bodyWorldPos = m_body->GetWorldSpacePosition(m_localBodyPos);
			Vector3D<Real> cableDir = m_worldConnectionPoint - bodyWorldPos;
			Real SqrdLength = cableDir.SquaredLength();

			if(SqrdLength < m_cableLength * m_cableLength)
				return false;

			SqrdLength = std::sqrt(SqrdLength);
			cableDir = cableDir / SqrdLength;

			newContact.m_body1 = m_body;
			newContact.m_contactNormal = cableDir;
			newContact.m_penetration = SqrdLength - m_cableLength;
			newContact.m_contactPoint = (bodyWorldPos);// + m_worldConnectionPoint) * 0.5f;
		
			return true;
		}

	private:
		Vector3D<Real> m_worldConnectionPoint;
	};

	class CableRigidBodyToRigidBody : public Cable
	{
	public:
		CableRigidBodyToRigidBody(
			RigidBody* body1, const Vector3D<Real>& localBodyPoint1,
			RigidBody* body2, const Vector3D<Real>& localBodyPoint2)
			: Cable(body1, localBodyPoint1),
			m_body2(body2),
			m_localBodyPos2(localBodyPoint2) {REQUIRES(body2)};
		~CableRigidBodyToRigidBody() {};

		virtual bool GetNewContact(Contact& newContact) const
		{
			const Vector3D<Real> bodyWorldPos1 = m_body->GetWorldSpacePosition(m_localBodyPos);
			const Vector3D<Real> bodyWorldPos2 = m_body2->GetWorldSpacePosition(m_localBodyPos2);
			Vector3D<Real> cableDir = bodyWorldPos1 - bodyWorldPos2;
			Real SqrdLength = cableDir.SquaredLength();

			if(SqrdLength < m_cableLength * m_cableLength)
				return false;

			SqrdLength = std::sqrt(SqrdLength);
			cableDir = cableDir / SqrdLength;

			newContact.m_body1 = m_body;
			newContact.m_contactNormal = cableDir;
			newContact.m_penetration = SqrdLength - m_cableLength;
			newContact.m_contactPoint = (bodyWorldPos1 + bodyWorldPos2) * 0.5f;
		
			return true;
		}

	private:
		RigidBody* m_body2;
		Vector3D<Real> m_localBodyPos2;
	};
}

#endif
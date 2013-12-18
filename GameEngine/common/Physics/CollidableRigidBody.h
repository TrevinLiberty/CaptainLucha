///////////////////////////////////////////////////////////////////////
//
//	Author: Trevin Liberty
//	Data:   8/26/2013
//	Description:
//		
//
///////////////////////////////////////////////////////////////////////

#ifndef COLLIDABLERIGIDBODY_H_CL
#define COLLIDABLERIGIDBODY_H_CL

#include "RigidBody.h"

namespace CaptainLucha
{
	class CollidableRigidBody : public CollisionListener, publi RigidBody
	{
	public:
		CollidableRigidBody();
		~CollidableRigidBody();

	protected:

	private:

	};
}

#endif
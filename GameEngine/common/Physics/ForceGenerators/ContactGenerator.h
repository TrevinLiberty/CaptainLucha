///////////////////////////////////////////////////////////////////////
//
//	Author: Trevin Liberty
//	Data:   9/4/2013
//	Description:
//		
//
///////////////////////////////////////////////////////////////////////

#ifndef CONTACTGENERATOR_H_CL
#define CONTACTGENERATOR_H_CL

#include "../RigidBody.h"
#include "../Contact.h"

namespace CaptainLucha
{
	class ContactGenerator
	{
	public:
		ContactGenerator() {};
		~ContactGenerator() {};

	virtual bool GetNewContact(Contact& newContact) const = 0;

	};
}

#endif
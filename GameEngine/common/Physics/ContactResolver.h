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
 *	@file	ContactResolver.h
 *	@brief	
 *
/****************************************************************************/

#ifndef CONTACTRESOLVER_H_CL
#define CONTACTRESOLVER_H_CL

#include "Contact.h"
#include "ForceGenerators/ContactGenerator.h"
#include "RigidBody.h"
#include "Primitives/Primitive.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class ContactResolver
	{
	public:
		ContactResolver(RigidBody* body1, Primitive* primitive1, RigidBody* body2, Primitive* primitive2);
		~ContactResolver();

		void GenerateContacts();

		void ResolveContacts();

		std::vector<Contact> m_contacts;

	protected:
		void InitContacts();
		void ResolvePosition();
		void ResolveVelocity();

		void UpdateBodyPenetration(
			RigidBody* body, Contact& contact, int sign, 
			const Vector3D<Real>& relativePos, const Vector3D<Real>& linearChange, 
			const Vector3D<Real>& angularChange);

	private:
		RigidBody* m_body1;
		RigidBody* m_body2;

		Primitive* m_primitive1;
		Primitive* m_primitive2;
	};
}

#endif
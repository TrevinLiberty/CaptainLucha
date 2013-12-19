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
 *	@file	SpringGenerator.h
 *	@brief	
 *  @attention Significant resource for implementation: http://procyclone.com/
 *
/****************************************************************************/

#ifndef SPRING_H_CL
#define SPRING_H_CL

#include "ForceGenerator.h"
#include "Renderer/RendererUtils.h"

namespace CaptainLucha
{
	class SpringGenerator : public CustomForceGenerator
	{
	public:
		void SetSpringConstant(float constant) {m_springConstant = constant;}
		void SetSpringRestLength(float length) {m_restLength = length;}

		virtual void Draw(GLProgram& glProgram) {}

	protected:
		Real m_springConstant;
		Real m_restLength;
	};

	class SpringRigidBodyToPoint : public SpringGenerator
	{
	public:
		SpringRigidBodyToPoint(RigidBody* body, const Vector3Df& localBodyPoint, const Vector3Df& worldConnectionPoint)
			: m_body(body),
			  m_localBodyPoint(localBodyPoint),
			  m_worldConnectionPoint(worldConnectionPoint)
		{REQUIRES(body)}

		~SpringRigidBodyToPoint() {};

		void UpdateForce(double DT)
		{
			Vector3D<Real> worldSpaceBodyPosition = m_body->GetWorldSpacePosition(m_localBodyPoint);

			Vector3D<Real> force = worldSpaceBodyPosition - m_worldConnectionPoint;

			Real length = force.Length();
			Real magnitude = length;
			magnitude = abs(magnitude - m_restLength);
			magnitude *= m_springConstant;

			force *= (1.0f / length);
			force *= -magnitude * DT;
			m_body->ApplyForceToWorldPoint(worldSpaceBodyPosition, force);
		}

		void Draw(GLProgram& glProgram)
		{
			SetGLProgram(&glProgram);
			DrawBegin(CL_LINES);
			{
				glLineWidth(3.0f);
				Vector3Df bodyPos = m_body->GetWorldSpacePosition(m_localBodyPoint);
				clVertex3(bodyPos.x, bodyPos.y, bodyPos.z);
				clVertex3(m_worldConnectionPoint.x, m_worldConnectionPoint.y, m_worldConnectionPoint.z);
			}
			DrawEnd();
			SetGLProgram(NULL);
			glLineWidth(1.0f);
		}

	private:
		RigidBody* m_body;

		Vector3Df m_localBodyPoint;
		Vector3Df m_worldConnectionPoint;
	};

	class SpringRigidBodyToRigidBody : public SpringGenerator
	{
	public:
		SpringRigidBodyToRigidBody(RigidBody* body1, const Vector3Df& localBodyPoint1, RigidBody* body2, const Vector3Df& localBodyPoint2)
			: m_body1(body1),
			  m_body2(body2),
			  m_localBodyPoint1(localBodyPoint1),
			  m_localBodyPoint2(localBodyPoint2)
		{
			REQUIRES(body1)
			REQUIRES(body2)
		}
		~SpringRigidBodyToRigidBody() {};

		void UpdateForce(double DT)
		{
			Vector3D<Real> worldSpaceBodyPosition1 = m_body1->GetWorldSpacePosition(m_localBodyPoint1);
			Vector3D<Real> worldSpaceBodyPosition2 = m_body2->GetWorldSpacePosition(m_localBodyPoint2);

			Vector3D<Real> force = worldSpaceBodyPosition1 - worldSpaceBodyPosition2;

			Real length = force.Length();
			Real magnitude = length;
			magnitude = abs(magnitude - m_restLength);
			magnitude *= m_springConstant;

			force *= (1.0f / length);
			force *= -magnitude * DT;
			trace(force)
			m_body1->ApplyForceToWorldPoint(worldSpaceBodyPosition1, force);
			m_body2->ApplyForceToWorldPoint(worldSpaceBodyPosition2, force * -1.0f);
		}

	protected:

	private:
		RigidBody* m_body1;
		RigidBody* m_body2;

		Vector3Df m_localBodyPoint1;
		Vector3Df m_localBodyPoint2;
	};
}

#endif
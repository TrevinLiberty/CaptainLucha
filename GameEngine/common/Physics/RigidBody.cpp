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

#include "RigidBody.h"

namespace CaptainLucha
{
	RigidBody::RigidBody()
		: m_isAwake(true),
		  m_invMass(0.0f),
		  m_linearDamp(1.0f),
		  m_angularDamp(1.0f),
		  m_transformMatrix(true),
		  m_avgLinearVelocity(SLEEP_SQRD_LINEAR_VELOCITY * 1.1f),
		  m_avgAngularVelocity(SLEEP_SQRD_ANGULAR_VELOCITY * 1.1f)
	{

	}

	RigidBody::~RigidBody()
	{

	}

	void RigidBody::Integrate(double DT)
	{
		if(!m_isAwake || FloatEquality(m_mass, 0.0f))
			return;

		m_lastFrameAcceleration = m_acceleration;
		m_lastFrameAcceleration += m_forceAccunulator * m_invMass;

		Vector3D<Real> angularAcceleration = m_invInertiaTensorWorld * m_torqueAccumulator;

		m_velocity += m_lastFrameAcceleration * DT;
		m_angularVelocity += angularAcceleration * DT;

		m_velocity *= pow(m_linearDamp, DT);
		m_angularVelocity *= pow(m_angularDamp, DT);

		m_position += m_velocity * DT;
		m_orientation.AddScaledVector(m_angularVelocity, DT);

		CalculatedDerivedData();
		UpdateSleepMode();
		ClearAccumulators();
	}

	void RigidBody::ApplyForce(const Vector3D<Real>& force)
	{
		m_forceAccunulator += force;
		m_isAwake = true;
	}

	void RigidBody::ApplyForceToWorldPoint(const Vector3D<Real>& worldPos, const Vector3D<Real>& force)
	{
		Vector3D<Real> position = worldPos;
		position -= GetPosition();

		m_forceAccunulator += force;
		m_torqueAccumulator += position.CrossProduct(force);

		m_isAwake = true;
	}

	void RigidBody::ApplyForceToLocalPoint(const Vector3D<Real>& localPos, const Vector3D<Real>& force)
	{
		Vector3D<Real> wsPos = GetWorldSpacePosition(localPos);
		ApplyForceToWorldPoint(localPos, force);
	}

	void RigidBody::SetAwakeState(bool val)
	{
		if(val && !FloatEquality(m_mass, 0.0f))
		{
			if(!m_isAwake)
			{
				m_avgLinearVelocity = SLEEP_SQRD_LINEAR_VELOCITY * 10.0;
				m_avgAngularVelocity = SLEEP_SQRD_ANGULAR_VELOCITY * 10.0;
			}
			m_isAwake = true;
		}
		else
		{
			m_isAwake = false;
			m_velocity = Vector3D<Real>();
			m_angularVelocity = Vector3D<Real>();
		}
	}

	void RigidBody::Wake()
	{
		m_avgLinearVelocity = SLEEP_SQRD_LINEAR_VELOCITY * 10.0;
		m_avgAngularVelocity = SLEEP_SQRD_ANGULAR_VELOCITY * 10.0;
		m_isAwake = true;
	}

	void RigidBody::SetInertiaTensorCuboid(Real width, Real height, Real depth, Real mass)
	{
		m_invInertiaTensor;

		Real t = mass * (1 / 12.0f);
		Real hh = height*height;
		Real ww = width*width;
		Real dd = depth*depth;

		m_invInertiaTensor[0] = t * (hh + dd);
		m_invInertiaTensor[4] = t * (ww + dd);
		m_invInertiaTensor[8] = t * (ww + hh);

		m_mass = mass;
		m_invMass = 1 / mass;
		m_invInertiaTensor = m_invInertiaTensor.GetInverse();
	}

	void RigidBody::SetInertiaTensorSphere(Real radius, Real mass)
	{
		Real t = mass * (2 / 5.0f) * radius * radius;

		m_invInertiaTensor[0] = t;
		m_invInertiaTensor[4] = t;
		m_invInertiaTensor[8] = t;

		m_mass = mass;
		m_invMass = 1 / mass;
		m_invInertiaTensor = m_invInertiaTensor.GetInverse();
	}

	void RigidBody::CalculateTransformation()
	{
		m_transformMatrix.MakeTransformation(GetPosition(), m_orientation);
	}

	void RigidBody::CalculatedDerivedData()
	{
		m_orientation.Normalize();
		CalculateTransformation();
		CalculateInvInteriaTensorWorld();
	}

	void RigidBody::CalculateInvInteriaTensorWorld()
	{
		Real t4 = m_transformMatrix.Data()[0]*m_invInertiaTensor.Data()[0]+
			m_transformMatrix.Data()[1]*m_invInertiaTensor.Data()[3]+
			m_transformMatrix.Data()[2]*m_invInertiaTensor.Data()[6];
		Real t9 = m_transformMatrix.Data()[0]*m_invInertiaTensor.Data()[1]+
			m_transformMatrix.Data()[1]*m_invInertiaTensor.Data()[4]+
			m_transformMatrix.Data()[2]*m_invInertiaTensor.Data()[7];
		Real t14 = m_transformMatrix.Data()[0]*m_invInertiaTensor.Data()[2]+
			m_transformMatrix.Data()[1]*m_invInertiaTensor.Data()[5]+
			m_transformMatrix.Data()[2]*m_invInertiaTensor.Data()[8];
		Real t28 = m_transformMatrix.Data()[4]*m_invInertiaTensor.Data()[0]+
			m_transformMatrix.Data()[5]*m_invInertiaTensor.Data()[3]+
			m_transformMatrix.Data()[6]*m_invInertiaTensor.Data()[6];
		Real t33 = m_transformMatrix.Data()[4]*m_invInertiaTensor.Data()[1]+
			m_transformMatrix.Data()[5]*m_invInertiaTensor.Data()[4]+
			m_transformMatrix.Data()[6]*m_invInertiaTensor.Data()[7];
		Real t38 = m_transformMatrix.Data()[4]*m_invInertiaTensor.Data()[2]+
			m_transformMatrix.Data()[5]*m_invInertiaTensor.Data()[5]+
			m_transformMatrix.Data()[6]*m_invInertiaTensor.Data()[8];
		Real t52 = m_transformMatrix.Data()[8]*m_invInertiaTensor.Data()[0]+
			m_transformMatrix.Data()[9]*m_invInertiaTensor.Data()[3]+
			m_transformMatrix.Data()[10]*m_invInertiaTensor.Data()[6];
		Real t57 = m_transformMatrix.Data()[8]*m_invInertiaTensor.Data()[1]+
			m_transformMatrix.Data()[9]*m_invInertiaTensor.Data()[4]+
			m_transformMatrix.Data()[10]*m_invInertiaTensor.Data()[7];
		Real t62 = m_transformMatrix.Data()[8]*m_invInertiaTensor.Data()[2]+
			m_transformMatrix.Data()[9]*m_invInertiaTensor.Data()[5]+
			m_transformMatrix.Data()[10]*m_invInertiaTensor.Data()[8];

		m_invInertiaTensorWorld.Data()[0] = t4*m_transformMatrix.Data()[0]+
			t9*m_transformMatrix.Data()[1]+
			t14*m_transformMatrix.Data()[2];
		m_invInertiaTensorWorld.Data()[1] = t4*m_transformMatrix.Data()[4]+
			t9*m_transformMatrix.Data()[5]+
			t14*m_transformMatrix.Data()[6];
		m_invInertiaTensorWorld.Data()[2] = t4*m_transformMatrix.Data()[8]+
			t9*m_transformMatrix.Data()[9]+
			t14*m_transformMatrix.Data()[10];
		m_invInertiaTensorWorld.Data()[3] = t28*m_transformMatrix.Data()[0]+
			t33*m_transformMatrix.Data()[1]+
			t38*m_transformMatrix.Data()[2];
		m_invInertiaTensorWorld.Data()[4] = t28*m_transformMatrix.Data()[4]+
			t33*m_transformMatrix.Data()[5]+
			t38*m_transformMatrix.Data()[6];
		m_invInertiaTensorWorld.Data()[5] = t28*m_transformMatrix.Data()[8]+
			t33*m_transformMatrix.Data()[9]+
			t38*m_transformMatrix.Data()[10];
		m_invInertiaTensorWorld.Data()[6] = t52*m_transformMatrix.Data()[0]+
			t57*m_transformMatrix.Data()[1]+
			t62*m_transformMatrix.Data()[2];
		m_invInertiaTensorWorld.Data()[7] = t52*m_transformMatrix.Data()[4]+
			t57*m_transformMatrix.Data()[5]+
			t62*m_transformMatrix.Data()[6];
		m_invInertiaTensorWorld.Data()[8] = t52*m_transformMatrix.Data()[8]+
			t57*m_transformMatrix.Data()[9]+
			t62*m_transformMatrix.Data()[10];
	}

	void RigidBody::ClearAccumulators()
	{
		m_forceAccunulator = Vector3D<Real>();
		m_torqueAccumulator = Vector3D<Real>();
	}

	void RigidBody::UpdateSleepMode()
	{
		const float BIAS = 0.5f;
		m_avgAngularVelocity = m_avgAngularVelocity * BIAS + m_angularVelocity.SquaredLength() * (1.0f - BIAS);
		m_avgLinearVelocity = m_avgLinearVelocity * BIAS + m_velocity.SquaredLength() * (1.0f - BIAS);

		if(m_avgAngularVelocity < SLEEP_SQRD_ANGULAR_VELOCITY
			&& m_avgLinearVelocity < SLEEP_SQRD_LINEAR_VELOCITY)
		{
			SetAwakeState(false);
		}
		else
		{
			if(m_avgLinearVelocity > 10 * SLEEP_SQRD_LINEAR_VELOCITY)
				m_avgLinearVelocity = 10 * SLEEP_SQRD_LINEAR_VELOCITY;
		
			if(m_avgAngularVelocity > 10 * SLEEP_SQRD_ANGULAR_VELOCITY)
				m_avgAngularVelocity = 10 * SLEEP_SQRD_ANGULAR_VELOCITY;
		}
	}

	Vector3D<Real> RigidBody::GetWorldSpacePosition(const Vector3D<Real>& pos)
	{
		return m_transformMatrix.TransformPosition(pos);
	}

	Real RigidBody::SLEEP_SQRD_LINEAR_VELOCITY = 0.1;
	Real RigidBody::SLEEP_SQRD_ANGULAR_VELOCITY = 0.1;
}
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
 *	@file	ExplosionGenerator.h
 *	@brief	
 *
/****************************************************************************/

#ifndef EXPLOSIONGENERATOR_H_CL
#define EXPLOSIONGENERATOR_H_CL

#include "ForceGenerator.h"

namespace CaptainLucha
{
	class ExplosionGenerator : public WorldForceGenerator
	{
	public:
		ExplosionGenerator()
			: m_timePassed(0.0f) {};
		~ExplosionGenerator() {};

		void FrameUpdate(double DT)
		{
			m_timePassed += static_cast<float>(DT);
		}

		void UpdateForce(RigidBody* body, double DT)
		{
			const Vector3D<Real>& rBodyPos = body->GetPosition();

			if(m_timePassed < m_implosionDurationSeconds)
			{
				const Real lengthSqrd = (rBodyPos - m_explosionPoint).SquaredLength();
				if(lengthSqrd >= m_implosionMinRadius*m_implosionMinRadius && lengthSqrd <= m_implosionMaxRadius*m_implosionMaxRadius)
				{
					Vector3D<Real> dir = m_explosionPoint - rBodyPos;
					dir.Normalize();
					body->ApplyForce(dir * m_implosionForce * body->GetMass());
				}
			}
			else if(m_timePassed < m_implosionDurationSeconds + m_shockwaveDurationSeconds)
			{
				const Real length = (rBodyPos - m_explosionPoint).Length();
				const Real shockTime = m_timePassed - m_implosionDurationSeconds;
				const Real currentMin = m_shockwaveSpeed * shockTime;
				const Real currentMax = currentMin + m_shockwaveThickness;
				const Real currentForce = (shockTime / m_shockwaveDurationSeconds) * m_peakForce;

				Vector3D<Real> appliedForce = (rBodyPos - m_explosionPoint) / length;

				if(currentMin - m_shockwaveThickness <= length && length < currentMin)
				{
					Real force = m_peakForce * (1 - (currentMin - length) / (m_shockwaveThickness));
					appliedForce *= force;
				}
				else if(currentMin <= length && length < currentMin + m_shockwaveThickness)
				{
					appliedForce *= m_peakForce;
				}
				else if(currentMin + m_shockwaveThickness <= length && length < currentMin + (1.0f + 1) * m_shockwaveThickness)
				{
					Real force = m_peakForce * (length - currentMin - m_shockwaveThickness) / m_shockwaveThickness;
					appliedForce *= force;
				}

				//body->ApplyForceToLocalPoint(Vector3Df(RandInRange(-0.5f, 0.5f), RandInRange(-0.5f, 0.5f), RandInRange(-0.5f, 0.5f)), appliedForce * body->GetMass());
				body->ApplyForce(appliedForce * body->GetMass());
				body->Wake();
			}
		}

		bool IsComplete()
		{
			return m_timePassed >= m_implosionDurationSeconds + m_shockwaveDurationSeconds;
		}

		void Reset()
		{
			m_timePassed = 0.0f;
		}

		Vector3D<Real> m_explosionPoint;

		//////////////////////////////////////////////////////////////////////////
		//	Implosion info
		Real m_implosionMaxRadius;
		Real m_implosionMinRadius;

		Real m_implosionDurationSeconds;
		
		Real m_implosionForce;

		//////////////////////////////////////////////////////////////////////////
		//	Shockwave info
		Real m_shockwaveSpeed;
		Real m_shockwaveThickness;
		Real m_peakForce;
		Real m_shockwaveDurationSeconds;

	private:
		Real m_timePassed;
	};
}

#endif
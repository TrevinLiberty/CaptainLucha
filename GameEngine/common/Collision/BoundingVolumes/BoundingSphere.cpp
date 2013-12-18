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

#include "BoundingSphere.h"
#include "AABoundingBox.h"

#include "Renderer/RendererUtils.h"

namespace CaptainLucha
{
	BoundingSphere::BoundingSphere(const BoundingSphere& rhs)
		 : BoundingVolume(CL_SPHERE)
	{
		m_pos = rhs.m_pos;
		m_radius = rhs.m_radius;
	}

	BoundingSphere::BoundingSphere(const std::vector<Vertex>& verts)
		 : BoundingVolume(CL_SPHERE)
	{
		AABoundingBox bb(verts);
		Vector3D<Real> mid = bb.Max() - bb.Min();

		m_radius = mid.Length() * 0.5f;
		m_pos = bb.Max() + bb.Min();
		m_pos /= 2.0f;
	}

	BoundingSphere::BoundingSphere(const std::vector<TangentSpaceVertex>& verts)
		 : BoundingVolume(CL_SPHERE)
	{
		AABoundingBox bb(verts);
		Vector3D<Real> mid = bb.Max() - bb.Min();

		m_radius = mid.Length() * 0.5f;
		m_pos = bb.Max() + bb.Min();
		m_pos /= 2.0f;
	}

	void BoundingSphere::Update(const Quaternion& rot, const Vector3D<Real>& trans)
	{
		UNUSED(rot)

		m_pos = trans;
	}

	AABoundingBox BoundingSphere::ConvertToAABB() const
	{
		return AABoundingBox(m_pos, Vector3D<Real>(m_radius, m_radius, m_radius));
	}

	Real BoundingSphere::SurfaceArea() const
	{
		return 4.0f * 3.141592f * m_radius * m_radius;
	}

	void BoundingSphere::DebugDraw() const
	{
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//DrawDebugSphere(m_pos, (float)m_radius);

		//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	bool BoundingSphere::BoundingSphereCollision(const BoundingSphere& other) const
	{
		Real sqrdLength = (m_pos - other.m_pos).SquaredLength();
		return sqrdLength <= (other.m_radius + m_radius) * (other.m_radius + m_radius);
	}
}

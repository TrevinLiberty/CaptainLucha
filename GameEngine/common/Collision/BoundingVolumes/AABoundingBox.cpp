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

#include "AABoundingBox.h"

#include "Renderer/RendererUtils.h"

#include <limits>

namespace CaptainLucha
{
	AABoundingBox::AABoundingBox(const std::vector<Vertex>& verts)
		: BoundingVolume(CL_AABB),
		  m_min(std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest()),
		  m_max(std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity())
	{
		CreateAABB(verts);
		m_baseMin = m_min;
		m_baseMax = m_max;
	}

	AABoundingBox::AABoundingBox(const std::vector<TangentSpaceVertex>& verts)
		: BoundingVolume(CL_AABB),
	      m_max(std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest()),
		  m_min(std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity())
	{
		CreateAABB(verts);
		m_baseMin = m_min;
		m_baseMax = m_max;
	}

	AABoundingBox::AABoundingBox(const Vector3D<Real>& pos, const Vector3D<Real>& extent)
		: BoundingVolume(CL_AABB)
	{
		m_min = pos - extent;
		m_max = pos + extent;
		m_baseMin = m_min;
		m_baseMax = m_max;
	}

	AABoundingBox::~AABoundingBox()
	{

	}

	AABoundingBox& AABoundingBox::operator=(const AABoundingBox& rhs)
	{
		m_min = rhs.m_min;
		m_max = rhs.m_max;
		m_baseMin = m_min;
		m_baseMax = m_max;
		return *this;
	}

	Vector3D<Real> AABoundingBox::GetCenter() const
	{
		Vector3D<Real> result = m_min + m_max;
		return result * 0.5f;
	}

	bool AABoundingBox::BoundingBoxCollision(const AABoundingBox& otherBox) const
	{
		if(m_max.x < otherBox.m_min.x || m_min.x > otherBox.m_max.x) return false;
		if(m_max.y < otherBox.m_min.y || m_min.y > otherBox.m_max.y) return false;
		if(m_max.z < otherBox.m_min.z || m_min.z > otherBox.m_max.z) return false;

		return true;
	}

	bool AABoundingBox::PointInBB(const Vector3Df& point) const
	{
		if(point.x > m_max.x || point.x < m_min.x) return false;
		if(point.y > m_max.y || point.y < m_min.y) return false;
		if(point.z > m_max.z || point.z < m_min.z) return false;

		return true;
	}

	void AABoundingBox::Update(const Quaternion& orientation, const Vector3D<Real>& trans)
	{
		Matrix3D<Real> rot = orientation.GetMatrix();

		for(int i = 0; i < 3; ++i)
		{
			m_min[i] = m_max[i] = trans[i];

			for(int j = 0; j < 3; ++j)
			{
				Real e = rot[i * 3 + j] * m_baseMin[j];
				Real f = rot[i * 3 + j] * m_baseMax[j];

				if(e < f)
				{
					m_min[i] += e;
					m_max[i] += f;
				}
				else
				{
					m_min[i] += f;
					m_max[i] += e;
				}
			}
		}
	}

	void AABoundingBox::CreateAABB(const std::vector<Vertex>& verts)
	{
		for(size_t i = 0; i < verts.size(); ++i)
		{
			if(verts[i].x_ < m_min.x)
				m_min.x = verts[i].x_;
			if(verts[i].y_ < m_min.y)
				m_min.y = verts[i].y_;
			if(verts[i].z_ < m_min.z)
				m_min.z = verts[i].z_;

			if(verts[i].x_ > m_max.x)
				m_max.x = verts[i].x_;
			if(verts[i].y_ > m_max.y)
				m_max.y = verts[i].y_;
			if(verts[i].z_ > m_max.z)
				m_max.z = verts[i].z_;
		}
	}

	void AABoundingBox::CreateAABB(const std::vector<TangentSpaceVertex>& verts)
	{
		for(size_t i = 0; i < verts.size(); ++i)
		{
			if(verts[i].x_ < m_min.x)
				m_min.x = verts[i].x_;
			if(verts[i].y_ < m_min.y)
				m_min.y = verts[i].y_;
			if(verts[i].z_ < m_min.z)
				m_min.z = verts[i].z_;

			if(verts[i].x_ > m_max.x)
				m_max.x = verts[i].x_;
			if(verts[i].y_ > m_max.y)
				m_max.y = verts[i].y_;
			if(verts[i].z_ > m_max.z)
				m_max.z = verts[i].z_;
		}
	}

	void AABoundingBox::CombineAABB(const AABoundingBox& lhs, const AABoundingBox& rhs)
	{
		m_min.x = min(lhs.m_min.x, rhs.m_min.x);
		m_min.y = min(lhs.m_min.y, rhs.m_min.y);
		m_min.z = min(lhs.m_min.z, rhs.m_min.z);

		m_max.x = max(lhs.m_max.x, rhs.m_max.x);
		m_max.y = max(lhs.m_max.y, rhs.m_max.y);
		m_max.z = max(lhs.m_max.z, rhs.m_max.z);
	}

	void AABoundingBox::Transform(const Matrix4Df& transform)
	{
		m_min = transform.TransformPosition(m_min);
		m_max = transform.TransformPosition(m_max);
	}

	void AABoundingBox::Transform(const Vector3Df& pos)
	{
		m_min += pos;
		m_max += pos;
	}

	Vector3Df AABoundingBox::GetRandomPosInside() const
	{
		return Vector3Df(
			(float)RandInRange(m_min.x, m_max.x), 
			(float)RandInRange(m_min.y, m_max.y), 
			(float)RandInRange(m_min.z, m_max.z));
	}

	void AABoundingBox::ResetBV()
	{
		m_min = m_baseMin;
		m_max = m_baseMax;
	}

	Real AABoundingBox::SurfaceArea() const
	{
		Vector3D<Real> size = m_max - m_min;
		return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
	}

	void AABoundingBox::DebugDraw() const
	{
		SetColor(Color::White);
		GetCurrentProgram()->SetUniform("emissive", 1.0f);

		DrawBegin(CL_LINES);
		clVertex3(m_min.x, m_min.y, m_min.z);
		clVertex3(m_min.x, m_min.y, m_max.z);
		clVertex3(m_min.x, m_min.y, m_min.z);
		clVertex3(m_max.x, m_min.y, m_min.z);
		clVertex3(m_min.x, m_min.y, m_max.z);
		clVertex3(m_max.x, m_min.y, m_max.z);
		clVertex3(m_max.x, m_min.y, m_min.z);
		clVertex3(m_max.x, m_min.y, m_max.z);

		clVertex3(m_min.x, m_max.y, m_min.z);
		clVertex3(m_min.x, m_max.y, m_max.z);
		clVertex3(m_min.x, m_max.y, m_min.z);
		clVertex3(m_max.x, m_max.y, m_min.z);
		clVertex3(m_min.x, m_max.y, m_max.z);
		clVertex3(m_max.x, m_max.y, m_max.z);
		clVertex3(m_max.x, m_max.y, m_min.z);
		clVertex3(m_max.x, m_max.y, m_max.z);

		clVertex3(m_min.x, m_min.y, m_min.z);
		clVertex3(m_min.x, m_max.y, m_min.z);
		clVertex3(m_min.x, m_min.y, m_max.z);
		clVertex3(m_min.x, m_max.y, m_max.z);
		clVertex3(m_max.x, m_min.y, m_min.z);
		clVertex3(m_max.x, m_max.y, m_min.z);
		clVertex3(m_max.x, m_min.y, m_max.z);
		clVertex3(m_max.x, m_max.y, m_max.z);
		DrawEnd();
	}
}
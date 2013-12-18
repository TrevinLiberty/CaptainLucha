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

#include "Rectangle.h"

#include <Utils/UtilDebug.h>
#include <algorithm>

namespace CaptainLucha
{
	Rectangle::Rectangle()
	{

	}

	Rectangle::Rectangle( const Rectangle& otherPoly )
	{
		*this = otherPoly;
	}

	Rectangle::~Rectangle()
	{
		//delete [] m_verts;
	}

	float Rectangle::GetBBArea() const
	{
		Vector2Df min = m_verts[0];
		Vector2Df max = m_verts[0];

		for(int i = 1; i < 4; ++i)
		{
			if(m_verts[i].x < min.x)
				min.x = m_verts[i].x;
			else if(m_verts[i].x > max.x)
				max.x = m_verts[i].x;

			if(m_verts[i].y < min.y)
				min.y = m_verts[i].y;
			else if(m_verts[i].y > max.y)
				max.y = m_verts[i].y;
		}

		return (max.x - min.x) * (max.y - min.y);
	}

	Vector2Df Rectangle::GetCenter() const
	{
		Vector2Df center;
		for(int i = 0; i < 4; ++i)
		{
			center += m_verts[i];
		}

		return center * 0.25;
	}

	void Rectangle::ReorderVerts()
	{
		std::vector<float> xs;
		std::vector<float> ys;

		for(int i = 0; i < 4; ++i)
		{
			xs.push_back(m_verts[i].x);
			ys.push_back(m_verts[i].y);
		}

		std::sort(xs.begin(), xs.end());
		std::sort(ys.begin(), ys.end());

		m_verts[0] = Vector2Df(xs[0], ys[0]);
		m_verts[1] = Vector2Df(xs[2], ys[0]);
		m_verts[2] = Vector2Df(xs[2], ys[2]);
		m_verts[3] = Vector2Df(xs[0], ys[2]);
	}

	Rectangle Rectangle::ExpandRectangle(float size)
	{
		Rectangle newRectangle;
		newRectangle[0] = Vector2Df(m_verts[0].x - size, m_verts[0].y - size);
		newRectangle[1] = Vector2Df(m_verts[1].x + size, m_verts[1].y - size);
		newRectangle[2] = Vector2Df(m_verts[2].x + size, m_verts[2].y + size);
		newRectangle[3] = Vector2Df(m_verts[3].x - size, m_verts[3].y + size);

		return newRectangle;
	}

	bool Rectangle::DoesOrderedRecContain(const Vector2Df& pos)
	{
		return pos.x >= m_verts[0].x && pos.x <= m_verts[2].x && pos.y >= m_verts[0].y && pos.y <= m_verts[2].y;
	}

	int Rectangle::GetNumSharedVerts(const Rectangle& poly)
	{
		int result = 0;
		for(int i = 0; i < 4; ++i)
		{
			for(int j = 0; j < 4; ++j)
			{
				if(m_verts[i] == poly.m_verts[j])
				{
					++result;
				}
			}
		}

		return result;
	}

	void Rectangle::GetSharedVerts(const Rectangle& poly, std::vector<Vector2Df>& sharedVerts)
	{
		for(int i = 0; i < 4; ++i)
		{
			for(int j = 0; j < 4; ++j)
			{
				if(m_verts[i] == poly.m_verts[j])
					sharedVerts.push_back(m_verts[i]);
			}
		}
	}

	void Rectangle::GetSharedVertsIndices(const Rectangle& poly, std::vector<int>& sharedVertsIndices)
	{
		for(int i = 0; i < 4; ++i)
		{
			for(int j = 0; j < 4; ++j)
			{
				if(m_verts[i] == poly.m_verts[j])
					sharedVertsIndices.push_back(i);
			}
		}
	}

	void Rectangle::GetSharedVertsIndices(const Rectangle& poly, std::vector<std::pair<int, int> >& sharedVertsIndices)
	{
		for(int i = 0; i < 4; ++i)
		{
			for(int j = 0; j < 4; ++j)
			{
				if(m_verts[i] == poly.m_verts[j])
					sharedVertsIndices.push_back(std::pair<int, int>(i, j));
			}
		}
	}

	void Rectangle::GetSharedVerts(const Rectangle& poly, std::vector<Vector2Df>& sharedVerts, std::vector<int>& sharedVertsIndices)
	{
		GetSharedVerts(poly, sharedVerts);
		GetSharedVertsIndices(poly, sharedVertsIndices);
	}

	int Rectangle::GetClosestVertIndexTo(const Vector2Df& point) const
	{
		int closest = 0;
		float distSqrd = (point - m_verts[0]).SquaredLength();

		for(int i = 1; i < 4; ++i)
		{
			if((point - m_verts[i]).SquaredLength() < distSqrd)
				closest = i;
		}

		return closest;
	}

	void Rectangle::GetMiddleOfEdges(std::vector<Vector2Df>& outEdges) const
	{
		outEdges.reserve(4);

		int j = 3;
		for(int i = 0; i < 4; ++i)
		{
			outEdges.push_back((m_verts[j] + m_verts[i]) * 0.5f);
			j = i;
		}
	}

	bool Rectangle::Contains(const Vector2Df& pos) const
	{
		int j = 3;
		bool result = false;
		for(int i = 0; i < 4; ++i)
		{
			if(m_verts[i].y > pos.y != m_verts[j].y > pos.y)
			{
				if(pos.x < (m_verts[j].x - m_verts[i].x) * (pos.y - m_verts[i].y) / (m_verts[j].y - m_verts[i].y) + m_verts[i].x)
					result = !result;
			}

			j = i;
		}

		return result;
	}

	bool Rectangle::Contains(const Rectangle& poly) const
	{
		for(int i = 0; i < 4; ++i)
		{
			if(!this->Contains(poly[i]))
				return false;
		}
		return true;
	}

	bool Rectangle::LineCollision(const Vector2Df& l1, const Vector2Df& l2)
	{
		int j = 3;
		for(int i = 0; i < 4; ++i)
		{
			if(DoLinesIntersect(l1, l2, m_verts[i], m_verts[j]))
				return true;

			j = i;
		}

		return false;
	}

	CollisionType Rectangle::ConvexCollides(const Rectangle& poly) const
	{
		Vector2Df axes1[4];
		Vector2Df axes2[4];

		GetAxes(axes1);
		for(size_t i = 0; i < 4; ++i)
		{
			Vector2Df proj1 = GetProjectionOnToAxis(axes1[i]);
			Vector2Df proj2 = poly.GetProjectionOnToAxis(axes1[i]);

			if(!ProjOverlap(proj1, proj2))
				return CL_None;
		}

		poly.GetAxes(axes2);
		for(size_t i = 0; i < 4; ++i)
		{
			Vector2Df proj1 = GetProjectionOnToAxis(axes2[i]);
			Vector2Df proj2 = poly.GetProjectionOnToAxis(axes2[i]);

			if(!ProjOverlap(proj1, proj2))
				return CL_None;
		}

		if(this->Contains(poly))
			return CL_Contains;
		else if(poly.Contains(*this))
			return CL_Inside;

		return CL_Collides;
	}

	void Rectangle::Subdivide(std::vector<Rectangle>& newPolys)
	{
		Vector2Df center;
		Vector2Df edgeMidpoints[4];

		for(int i = 0; i < 4; ++i)
		{
			center += m_verts[i];
		}

		center *= 0.25;

		int j = 3;
		for(int i = 0; i < 4; ++i)
		{
			edgeMidpoints[i] = (m_verts[i] + m_verts[j]) * 0.5f;
			j = i;
		}

		newPolys.push_back(Rectangle());
		newPolys.back()[0] = m_verts[0];
		newPolys.back()[1] = edgeMidpoints[1];
		newPolys.back()[2] = center;
		newPolys.back()[3] = edgeMidpoints[0];

		newPolys.push_back(Rectangle());
		newPolys.back()[0] = edgeMidpoints[1];
		newPolys.back()[1] = m_verts[1];
		newPolys.back()[2] = edgeMidpoints[2];
		newPolys.back()[3] = center;

		newPolys.push_back(Rectangle());
		newPolys.back()[0] = center;
		newPolys.back()[1] = edgeMidpoints[2];
		newPolys.back()[2] = m_verts[2];
		newPolys.back()[3] = edgeMidpoints[3];

		newPolys.push_back(Rectangle());
		newPolys.back()[0] = edgeMidpoints[0];
		newPolys.back()[1] = center;
		newPolys.back()[2] = edgeMidpoints[3];
		newPolys.back()[3] = m_verts[3];
	}

	const Vector2Df& Rectangle::GetPrevVert(int index)
	{
		int result = index - 1;
		if(result < 0)
			result = 3;

		return m_verts[result];
	}

	const Vector2Df& Rectangle::GetNextVert(int index)
	{

		int result = index + 1;
		if(result == 4)
			result = 0;

		return m_verts[result];
	}

	int Rectangle::GetPrevVertIndex(int index)
	{
		int result = index - 1;
		if(result < 0)
			result = 3;

		return result;
	}

	int Rectangle::GetNextVertIndex(int index)
	{
		int result = index + 1;
		if(result == 4)
			result = 0;

		return result;
	}

	void Rectangle::operator=( const Rectangle& otherPoly )
	{
		memcpy(m_verts, otherPoly.m_verts, 4 * sizeof(Vector2Df));
	}

	Rectangle Rectangle::operator+(const Vector2Df& dir) const
	{
		Rectangle result = *this;
		for(int i = 0; i < 4; ++i)
		{
			result.m_verts[i] += dir;
		}

		return result;
	}

	Rectangle Rectangle::operator-(const Vector2Df& dir) const
	{
		Rectangle result = *this;
		for(int i = 0; i < 4; ++i)
		{
			result.m_verts[i] -= dir;
		}

		return result;
	}

	Rectangle& Rectangle::operator+=(const Vector2Df& dir)
	{
		for(int i = 0; i < 4; ++i)
		{
			m_verts[i] += dir;
		}

		return *this;
	}

	Rectangle& Rectangle::operator-=(const Vector2Df& dir)
	{
		for(int i = 0; i < 4; ++i)
		{
			m_verts[i] -= dir;
		}

		return *this;
	}

	void Rectangle::DebugTracePoly()
	{
		for(int i = 0; i < 4; ++i)
			trace(m_verts[i].x << " " << m_verts[i].y)
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void Rectangle::GetAxes(Vector2Df* axes) const
	{
		int j = 3;
		for(int i = 0; i < 4; ++i)
		{
			axes[i] = (m_verts[i] - m_verts[j]).Perp();
			axes[i].FastNormalize();
			j = i;
		}
	}

	Vector2Df Rectangle::GetProjectionOnToAxis(const Vector2Df& axis) const
	{
		float min = axis.Dot(m_verts[0]);
		float max = min;

		for(int i = 1; i < 4; ++i)
		{
			float temp = axis.Dot(m_verts[i]);
			if(temp < min)
				min = temp;
			else if(temp > max)
				max = temp;
		}

		return Vector2Df(min, max);
	}

	bool Rectangle::ProjOverlap(const Vector2Df& a, const Vector2Df& b) const
	{
		if(a.x > b.x && a.x < b.y)
			return true;
		
		if(a.y > b.x && a.y < b.y)
			return true;

		if(b.x > a.x && b.x < a.y)
			return true;

		if(b.y > a.x && b.y < a.y)
			return true;

		return false;
	}
}
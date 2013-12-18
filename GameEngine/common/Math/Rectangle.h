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
 *	@file	Rectangle.h
 *	@brief	
 *
/****************************************************************************/

#ifndef POLYGON_H_CL
#define POLYGON_H_CL

#include <Math/Vector2D.h>

namespace CaptainLucha
{
	enum CollisionType
	{
		CL_None,
		CL_Collides,
		CL_Contains,
		CL_Inside
	};

	class Rectangle
	{
	public:
		Rectangle();
		Rectangle(const Rectangle& otherPoly);
		~Rectangle();

		const Vector2Df* Verts() const { return m_verts; }
		Vector2Df* Verts() { return m_verts; }

		const Vector2Df& operator[](int index) const {return m_verts[index];}
		Vector2Df& operator[](int index) {return m_verts[index];}

		const Vector2Df& GetNextVert(int index);
		const Vector2Df& GetPrevVert(int index);

		int GetNextVertIndex(int index);
		int GetPrevVertIndex(int index);

		float GetBBArea() const;

		Vector2Df GetCenter() const;

		void ReorderVerts();
		Rectangle ExpandRectangle(float size);
		bool DoesOrderedRecContain(const Vector2Df& pos);

		int GetNumSharedVerts(const Rectangle& poly);
		void GetSharedVerts(const Rectangle& poly, std::vector<Vector2Df>& sharedVerts);
		void GetSharedVertsIndices(const Rectangle& poly, std::vector<int>& sharedVertsIndices);
		void GetSharedVertsIndices(const Rectangle& poly, std::vector<std::pair<int, int> >& sharedVertsIndices);
		void GetSharedVerts(const Rectangle& poly, std::vector<Vector2Df>& sharedVerts, std::vector<int>& sharedVertsIndices);

		int GetClosestVertIndexTo(const Vector2Df& point) const;

		void GetMiddleOfEdges(std::vector<Vector2Df>& outEdges) const;

		bool Contains(const Vector2Df& pos) const;
		bool Contains(const Rectangle& poly) const;

		bool LineCollision(const Vector2Df& l1, const Vector2Df& l2);

		CollisionType ConvexCollides(const Rectangle& poly) const;

		void Subdivide(std::vector<Rectangle>& newPolys);

		void operator=(const Rectangle& otherPoly);

		Rectangle operator+(const Vector2Df& dir) const;
		Rectangle operator-(const Vector2Df& dir) const;

		Rectangle& operator+=(const Vector2Df& dir);
		Rectangle& operator-=(const Vector2Df& dir);

		void DebugTracePoly();

	protected:
		void GetAxes(Vector2Df* axes) const;
		Vector2Df GetProjectionOnToAxis(const Vector2Df& axis) const;

		bool ProjOverlap(const Vector2Df& a, const Vector2Df& b) const;

	private:
		Vector2Df m_verts[4];
	};
}

#endif
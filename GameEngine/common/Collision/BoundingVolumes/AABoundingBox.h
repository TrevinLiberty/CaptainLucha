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
 *	@file	AABoundingBox.h
 *	@brief	
 *  @see CollisionFunctions.h CaptainLucha::BoundingVolumeCollision
 *	@todo	Overload operators
 *
/****************************************************************************/

#ifndef BOUNDING_AABB_H_CL
#define BOUNDING_AABB_H_CL

#include "Math/Vector3D.h"
#include "Math/Matrix3D.h"
#include "Math/Quaternion.h"
#include "BoundingVolume.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	struct TangentSpaceVertex;
	struct Vertex;

	/**
	* @brief     Holds min and max verts that define a AABB. 
	*Base min and max verts are saved to easily update the AABB based on some orientation.
	*/
	class AABoundingBox : public BoundingVolume
	{
	public:
		AABoundingBox()
			: BoundingVolume(CL_AABB) {};
		AABoundingBox(const AABoundingBox& rhs)
			: BoundingVolume(CL_AABB), m_min(rhs.m_min), m_max(rhs.m_max) {}
		AABoundingBox(const std::vector<Vertex>& verts);
		AABoundingBox(const std::vector<TangentSpaceVertex>& verts);
		AABoundingBox(const Vector3D<Real>& pos, const Vector3D<Real>& extent);
		~AABoundingBox();

		AABoundingBox& operator=(const AABoundingBox& rhs);

		const Vector3D<Real>& Min() const {return m_min;}
		const Vector3D<Real>& Max() const {return m_max;}

		/**
		 * @brief     Returns the middle of min and max
		 * @return    Vector3D<Real>
		 */
		Vector3D<Real> GetCenter() const;

		/**
		 * @brief     Determines if the boxes are colliding
		 * @param	  const AABoundingBox & otherBox
		 * @return    bool
		 */
		bool BoundingBoxCollision(const AABoundingBox& otherBox) const;

		bool PointInBB(const Vector3Df& point) const;

		/**
		 * @brief     Updates the AABB to a new position. 
		 *Updates the AABB size to encapsulate the oriented bounding box created from rot.
		 * @param	  const Quaternion & rot
		 * @param	  const Vector3D<Real> & trans

		 */
		void Update(const Quaternion& rot, const Vector3D<Real>& trans);

		/**
		 * @brief     Creates a AABB that encapsulates every vert
		 * @param	  const std::vector<Vertex> & verts

		 */
		void CreateAABB(const std::vector<Vertex>& verts);


		/**
		 * @brief     Creates a AABB that encapsulates every vert
		 * @param	  const std::vector<TangentSpaceVertex> & verts

		 */
		void CreateAABB(const std::vector<TangentSpaceVertex>& verts);

		/**
		 * @brief     Combines lhs and rhs to create a new AABB that encapsulates lhs and rhs.
		 * @param	  const AABoundingBox & lhs
		 * @param	  const AABoundingBox & rhs

		 */
		void CombineAABB(const AABoundingBox& lhs, const AABoundingBox& rhs);

		virtual AABoundingBox ConvertToAABB() const {return *this;}

		void Transform(const Matrix4Df& transform);
		void Transform(const Vector3Df& pos);

		Vector3Df GetRandomPosInside() const;

		void ResetBV();

		/**
		 * @brief     Returns the surface area of the AABB
		 * @return    Real
		 */
		Real SurfaceArea() const;

		/**
		 * @brief     Draws a white Cube. Slow

		 */
		virtual void DebugDraw() const;

	protected:

	private:
		Vector3D<Real> m_min;
		Vector3D<Real> m_max;

		Vector3D<Real> m_baseMin;
		Vector3D<Real> m_baseMax;
	};
}

#endif
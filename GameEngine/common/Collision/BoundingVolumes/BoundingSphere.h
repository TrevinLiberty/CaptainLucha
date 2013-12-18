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
 *	@file	BoundingSphere.h
 *	@brief	
 *  @see CollisionFunctions.h CaptainLucha::BoundingVolumeCollision
 *  @todo Overload operators
 *
/****************************************************************************/

#ifndef BOUNDINGSPHERE_H_CL
#define BOUNDINGSPHERE_H_CL

#include "BoundingVolume.h"
#include "Math/Vector3D.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class Quaternion;
	struct TangentSpaceVertex;
	struct Vertex;

	/**
	* @brief     Holds position and radius of a BoundingSphere
	*/
	class BoundingSphere : public BoundingVolume
	{
	public:
		BoundingSphere() : BoundingVolume(CL_SPHERE) {};
		BoundingSphere(const Vector3D<Real>& pos, Real radius)
			: BoundingVolume(CL_SPHERE), m_pos(pos), m_radius(radius) {};
		BoundingSphere(const BoundingSphere& rhs);
		BoundingSphere(const std::vector<Vertex>& verts);
		BoundingSphere(const std::vector<TangentSpaceVertex>& verts);
		~BoundingSphere() {};

		/**
		 * @brief     Updates the Sphere's position
		 * @param	  const Quaternion & rot Unused
		 * @param	  const Vector3D<Real> & trans
		 */
		void Update(const Quaternion& rot, const Vector3D<Real>& trans);

		const Vector3D<Real>& GetPosition() const {return m_pos;}
		Real GetRadius() const {return m_radius;}

		/**
		 * @brief     Creates a AABB that encapsulates the Sphere
		 * @return    CaptainLucha::AABoundingBox
		 */
		virtual AABoundingBox ConvertToAABB() const;

		/**
		 * @brief     Returns the surface area of the sphere
		 * @return    Real
		 */
		virtual Real SurfaceArea() const;

		/**
		 * @brief     Draws the sphere with lines

		 */
		virtual void DebugDraw() const;

		/**
		 * @brief     Does a radius test against other to determine collision
		 * @param	  const BoundingSphere & other
		 * @return    bool
		 */
		bool BoundingSphereCollision(const BoundingSphere& other) const;

	protected:

	private:
		Vector3D<Real> m_pos;
		Real m_radius;
	};
}

#endif
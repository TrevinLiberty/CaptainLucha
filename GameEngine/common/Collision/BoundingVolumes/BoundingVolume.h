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
 *	@file	BoundingVolume.h
 *	@brief	
 *  @see CollisionFunctions.h CaptainLucha::BoundingVolumeCollision
 *
/****************************************************************************/

#ifndef BOUNDING_VOLUME_H_CL
#define BOUNDING_VOLUME_H_CL

#include "Math/Vector3D.h"
#include "Math/Quaternion.h"

namespace CaptainLucha
{
	class AABoundingBox;

	enum BVType
	{
		CL_AABB,
		CL_SPHERE
	};

	class BoundingVolume
	{
	public:
		BoundingVolume(BVType type) : m_type(type) {}
		virtual ~BoundingVolume() {};

		/**
		 * @brief     Pure virtual for updating a BoundingVolume by a new rotation and translation
		 * @pure
		 * @param	  const Quaternion & rot
		 * @param	  const Vector3D<Real> & trans

		 */
		virtual void Update(const Quaternion& rot, const Vector3D<Real>& trans) = 0;

		/**
		 * @brief     Pure virtual for converting BoundingVolume to a AABB that encapsulates the BV.
		 * @pure	  
		 * @return    CaptainLucha::AABoundingBox
		 */
		virtual AABoundingBox ConvertToAABB() const = 0;

		/**
		 * @brief     Pure virtual for returning the BoundingVolume's sruface area
		 * @pure
		 * @return    Real
		 */
		virtual Real SurfaceArea() const = 0;

		/**
		 * @brief     Returns the BoundingVolume's type.
		 * @see		  BVType
		 * @return    CaptainLucha::BVType
		 */
		BVType GetType() const {return m_type;}

		/**
		 * @brief     Pure virtual for Drawing Debug

		 * @bug		  DeferredRenderer doesn't support debug drawing atm
		 */
		virtual void DebugDraw() const {};

	protected:
		BVType m_type;

	private:
	};
}

#endif
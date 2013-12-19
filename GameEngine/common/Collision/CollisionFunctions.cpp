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
 *  @attention Significant resource for implementation: http://procyclone.com/
 *
/****************************************************************************/

#include "CollisionFunctions.h"

#include "BoundingVolumes/AABoundingBox.h"
#include "BoundingVolumes/BoundingSphere.h"

#include "Physics/RigidBody.h"
#include "Physics/Primitives/Primitive_Box.h"
#include "Physics/Primitives/Primitive_Sphere.h"
#include "Physics/Primitives/Primitive_Plane.h"

#include "Utils/UtilDebug.h"

#include <algorithm>

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//BV Collision
	//////////////////////////////////////////////////////////////////////////
	bool Collides(const AABoundingBox& aabb1, const AABoundingBox& aabb2);
	bool Collides(const BoundingSphere& bs1, const BoundingSphere& bs2);

	bool Collides(const BoundingSphere& bs1, const AABoundingBox& aabb2);
	bool Collides(const AABoundingBox& aabb1, const BoundingSphere& bs2);

	bool BoundingVolumeCollision(const BoundingVolume* bv1, const BoundingVolume* bv2)
	{
		bool result = false;

		const AABoundingBox* aabb1 = bv1->GetType() == CL_AABB ? reinterpret_cast<const AABoundingBox*>(bv1) : NULL;
		const AABoundingBox* aabb2 = bv2->GetType() == CL_AABB ? reinterpret_cast<const AABoundingBox*>(bv2) : NULL;

		const BoundingSphere* bs1 = bv1->GetType() == CL_SPHERE ? reinterpret_cast<const BoundingSphere*>(bv1) : NULL;
		const BoundingSphere* bs2 = bv2->GetType() == CL_SPHERE ? reinterpret_cast<const BoundingSphere*>(bv2) : NULL;

		if(aabb1 != NULL && aabb2 != NULL)
			result = Collides(*aabb1, *aabb2);
		else if(bs1 != NULL && bs2 != NULL)
			result = Collides(*bs1, *bs2);
		else if(bs1 != NULL && aabb2 != NULL)
			result = Collides(*bs1, *aabb2);
		else if(aabb1 != NULL && bs2 != NULL)
			result = Collides(*aabb1, *bs2);

		return result;
	}

	bool Collides(const AABoundingBox& aabb1, const AABoundingBox& aabb2)
	{
		return aabb1.BoundingBoxCollision(aabb2);
	}

	bool Collides(const BoundingSphere& bs1, const BoundingSphere& bs2)
	{
		return bs1.BoundingSphereCollision(bs2);
	}

	bool Collides(const BoundingSphere& bs1, const AABoundingBox& aabb2)
	{
		Vector3D<Real> boxCenter = aabb2.GetCenter();
		Vector3D<Real> dirToSphere = bs1.GetPosition() - boxCenter;
		Vector3D<Real> closestBoxPointToSphere;

		Vector3D<Real> boxDim = (aabb2.Max() - aabb2.Min()) * 0.5f;

		if (dirToSphere.x < -boxDim.x)
			closestBoxPointToSphere.x = -boxDim.x;
		else if (dirToSphere.x > boxDim.x)
			closestBoxPointToSphere.x = boxDim.x;
		else
			closestBoxPointToSphere.x = dirToSphere.x;

		if (dirToSphere.y < -boxDim.y)
			closestBoxPointToSphere.y = -boxDim.y;
		else if (dirToSphere.y > boxDim.y)
			closestBoxPointToSphere.y = boxDim.y;
		else
			closestBoxPointToSphere.y = dirToSphere.y;

		if (dirToSphere.z < -boxDim.z)
			closestBoxPointToSphere.z = -boxDim.z;
		else if (dirToSphere.z > boxDim.z)
			closestBoxPointToSphere.z = boxDim.z;
		else
			closestBoxPointToSphere.z = dirToSphere.z;

		Vector3D<Real> dist = closestBoxPointToSphere - dirToSphere;

		return dist.SquaredLength() < bs1.GetRadius() * bs1.GetRadius();
	}

	bool Collides(const AABoundingBox& aabb1, const BoundingSphere& bs2)
	{
		return Collides(bs2, aabb1);
	}

	//////////////////////////////////////////////////////////////////////////
	//	Generate Contacts
	//////////////////////////////////////////////////////////////////////////
	bool GenerateContactsSphereSphere(const Primitive_Sphere* p1, const Primitive_Sphere* p2, std::vector<Contact>& outContact)
	{
		Vector3D<Real> pos1 = p1->GetTransformation().GetPosition();
		Vector3D<Real> pos2 = p2->GetTransformation().GetPosition();

		Real radius1 = p1->GetRadius();
		Real radius2 = p2->GetRadius();

		Vector3D<Real> dirToP1 = pos1 - pos2;
		Real length = dirToP1.Length();

		//Fast out?
		if(length <= 0.0f || length >= radius1*radius2)
		{
			return false;
		}

		outContact.push_back(Contact());
		auto& newContact = outContact.back();
		newContact.m_contactNormal = dirToP1 * (1.0f / length);
		newContact.m_contactPoint = pos1 - dirToP1 * 0.5f;
		newContact.m_penetration = (radius1+radius2) - length;

		return true;
	}

	bool GenerateContactsSpherePlane(const Primitive_Sphere* p1, const Primitive_Plane* p2, std::vector<Contact>& outContact)
	{
		const Vector3D<Real>& spherePos = p1->GetTransformation().GetPosition();

		Real length = p2->GetNormal().Dot(spherePos) - p1->GetRadius() - p2->GetOffset();

		if(length >= 0)
			return false;

		outContact.push_back(Contact());
		auto& newContact = outContact.back();
		newContact.m_contactNormal = p2->GetNormal();
		newContact.m_penetration = -length;
		newContact.m_contactPoint = spherePos - p2->GetNormal() * (length + p1->GetRadius());

		return true;
	}

	bool GenerateContactsBoxPlane(const Primitive_Box* box, const Primitive_Plane* plane, std::vector<Contact>& outContact)
	{
		const Vector3D<Real>& boxExtent = box->GetExtent();

		Vector3D<Real> vertices[8] =
		{
			Vector3D<Real>(-boxExtent.x, -boxExtent.y, -boxExtent.z),
			Vector3D<Real>(-boxExtent.x, -boxExtent.y, +boxExtent.z),
			Vector3D<Real>(-boxExtent.x, +boxExtent.y, -boxExtent.z),
			Vector3D<Real>(-boxExtent.x, +boxExtent.y, +boxExtent.z),
			Vector3D<Real>(+boxExtent.x, -boxExtent.y, -boxExtent.z),
			Vector3D<Real>(+boxExtent.x, -boxExtent.y, +boxExtent.z),
			Vector3D<Real>(+boxExtent.x, +boxExtent.y, -boxExtent.z),
			Vector3D<Real>(+boxExtent.x, +boxExtent.y, +boxExtent.z)
		};

		const size_t START_SIZE = outContact.size();
		for(int i = 0; i < 8; ++i)
		{
			vertices[i] = box->GetTransformation().TransformPosition(vertices[i]);

			Real length = vertices[i].Dot(plane->GetNormal());

			if(length <= plane->GetOffset())
			{
				outContact.push_back(Contact());
				auto& newContact = outContact.back();
				newContact.m_contactNormal = plane->GetNormal();
				newContact.m_contactPoint = plane->GetNormal();
				newContact.m_contactPoint *= length - plane->GetOffset();
				newContact.m_contactPoint += vertices[i];
				newContact.m_penetration = plane->GetOffset() - length;
			}
		}

		return START_SIZE != outContact.size();
	}

	bool GenerateContactsBoxSphere(const Primitive_Box* box, const Primitive_Sphere* sphere, std::vector<Contact>& outContact)
	{
		Vector3D<Real> spherePos = sphere->GetTransformation().GetPosition();
		Vector3D<Real> boxSpaceSpherePos = box->GetTransformation().TransformTransposePosition(spherePos);

		const Vector3D<Real>& boxExtent = box->GetExtent();
		Vector3D<Real> closestPoint(0.0f, 0.0f, 0.0f);
		Real length;

		length = boxSpaceSpherePos.x;
		if(length > boxExtent.x)
			length = boxExtent.x;
		else if(length < -boxExtent.x)
			length = -boxExtent.x;
		closestPoint.x = length;

		length = boxSpaceSpherePos.y;
		if(length > boxExtent.y)
			length = boxExtent.y;
		else if(length < -boxExtent.y)
			length = -boxExtent.y;
		closestPoint.y = length;

		length = boxSpaceSpherePos.z;
		if(length > boxExtent.z)
			length = boxExtent.z;
		else if(length < -boxExtent.z)
			length = -boxExtent.z;
		closestPoint.z = length;

		length = (closestPoint - boxSpaceSpherePos).SquaredLength();
		if(length > sphere->GetRadius() * sphere->GetRadius())
		{
			return false;
		}

		Vector3D<Real> closestPointWorld = box->GetTransformation().TransformPosition(closestPoint);

		outContact.push_back(Contact());
		auto& newContact = outContact.back();

		newContact.m_contactNormal = (closestPointWorld - spherePos);
		newContact.m_contactNormal.Normalize();
		newContact.m_contactPoint = closestPointWorld;
		newContact.m_penetration = sphere->GetRadius() - std::sqrt(length);
		return true;
	}

	inline Real TransformToAxis(const Primitive_Box* box, const Vector3D<Real>& axis)
	{
		const Vector3D<Real>& extent = box->GetExtent();
		Matrix3D<Real> rot = box->GetTransformation().GetOrientation().GetMatrix();
		return 
			extent.x * std::abs(axis.Dot(rot.GetAxis(0))) +
			extent.y * std::abs(axis.Dot(rot.GetAxis(1))) +
			extent.z * std::abs(axis.Dot(rot.GetAxis(2)));
	}

	inline Real GetAmountOfPenOnAxis(const Primitive_Box* box1, const Primitive_Box* box2, const Vector3D<Real>& axis, const Vector3D<Real>& dirToBox1)
	{
		Real project1 = TransformToAxis(box1, axis);
		Real project2 = TransformToAxis(box2, axis);

		Real length = abs(dirToBox1.Dot(axis));
		return (project1 + project2) - length;
	}

	void PointFaceContact(const Primitive_Box* box1, const Primitive_Box* box2, const Vector3D<Real>& dirToBox1, Contact& data, int bestAxisIndex, Real penetration)
	{
		Matrix3D<Real> rotation1 = box1->GetTransformation().GetOrientation().GetMatrix();
		Matrix3D<Real> rotation2 = box2->GetTransformation().GetOrientation().GetMatrix();

		Vector3D<Real> normal(rotation1.GetAxis(bestAxisIndex));
		if(normal.Dot(dirToBox1) > 0)
			normal *= -1.0f;

		Vector3D<Real> vertex = box2->GetExtent();
		if (rotation2.GetAxis(0).Dot(normal) < 0) 
			vertex.x = -vertex.x;    
		if (rotation2.GetAxis(1).Dot(normal) < 0) 
			vertex.y = -vertex.y;   
		if (rotation2.GetAxis(2).Dot(normal) < 0) 
			vertex.z = -vertex.z;

		data.m_contactNormal = normal;
		data.m_penetration = penetration;

		data.m_contactPoint = box2->GetTransformation().TransformPosition(vertex);
	}

	Vector3D<Real> CalculateContactPoint(
		const Vector3D<Real>& pOne, const Vector3D<Real>& dOne, Real oneSize, 
		const Vector3D<Real>& pTwo, const Vector3D<Real>& dTwo, Real twoSize, bool useOne)
	{
		Vector3D<Real> toSt, cOne, cTwo;
		Real dpStaOne, dpStaTwo, dpOneTwo, smOne, smTwo;
		Real denom, mua, mub;

		smOne = dOne.SquaredLength();
		smTwo = dTwo.SquaredLength();
		dpOneTwo = dTwo.Dot(dOne);

		toSt = pOne - pTwo;
		dpStaOne = dOne.Dot(toSt);
		dpStaTwo = dTwo.Dot(toSt);

		denom = smOne * smTwo - dpOneTwo * dpOneTwo;
		// Zero denominator indicates parrallel lines
		if (std::abs(denom) < 0.001f) 
		{        
			return useOne ? pOne : pTwo;    
		}    

		mua = (dpOneTwo * dpStaTwo - smTwo * dpStaOne) / denom;
		mub = (smOne * dpStaTwo - dpOneTwo * dpStaOne) / denom;

		// If either of the edges has the nearest point out
		// of bounds, then the edges aren't crossed, we have
		// an edge-face contact. Our point is on the edge, which
		// we know from the useOne parameter.
		if (mua > oneSize ||
			mua < -oneSize ||
			mub > twoSize ||
			mub < -twoSize)
		{        
			return useOne?pOne:pTwo;
		}    
		else    
		{        
			cOne = pOne + dOne * mua;
			cTwo = pTwo + dTwo * mub;
			return cOne * 0.5 + cTwo * 0.5;
		}
	}

	bool GenerateContactsBoxBox(const Primitive_Box* box1, const Primitive_Box* box2, std::vector<Contact>& outContact)
	{
		auto& box1Transorm = box1->GetTransformation();
		auto& box2Transorm = box2->GetTransformation();

		Vector3D<Real> toCenter = box2Transorm.GetPosition() - box1Transorm.GetPosition();

		Real smallestPen = std::numeric_limits<Real>::infinity();
		unsigned int bestCase = 0xffffffff;

		Matrix3D<Real> rotation1 = box1Transorm.GetOrientation().GetMatrix();
		Matrix3D<Real> rotation2 = box2Transorm.GetOrientation().GetMatrix();

		const Vector3D<Real>& axis11 = rotation1.GetAxis(0);
		const Vector3D<Real>& axis12 = rotation1.GetAxis(1);
		const Vector3D<Real>& axis13 = rotation1.GetAxis(2);

		const Vector3D<Real>& axis21 = rotation2.GetAxis(0);
		const Vector3D<Real>& axis22 = rotation2.GetAxis(1);
		const Vector3D<Real>& axis23 = rotation2.GetAxis(2);

		Vector3D<Real> axes[15];
		// Face axes for object one.
		axes[0] = axis11;
		axes[1] = axis12;
		axes[2] = axis13;
		// Face axes for object two.
		axes[3] = axis21;
		axes[4] = axis22;
		axes[5] = axis23;
		// Edge-edge axes.
		axes[6]  = axis11.CrossProduct(axis21);
		axes[7]  = axis11.CrossProduct(axis22);
		axes[8]  = axis11.CrossProduct(axis23);
		axes[9]  = axis12.CrossProduct(axis21);
		axes[10] = axis12.CrossProduct(axis22);
		axes[11] = axis12.CrossProduct(axis23);
		axes[12] = axis13.CrossProduct(axis21);
		axes[13] = axis13.CrossProduct(axis22);
		axes[14] = axis13.CrossProduct(axis23);

		int bestSingleAxis = 0;
		for(int i = 0; i < 15; ++i)
		{
			Vector3D<Real>& currentAxis = axes[i];

			if(currentAxis.SquaredLength() < 0.001)
				continue;
			currentAxis.Normalize();

			Real overlap = GetAmountOfPenOnAxis(box1, box2, currentAxis, toCenter);
			if(overlap < 0)
				return false;
			if(overlap < smallestPen)
			{
				smallestPen = overlap;
				bestCase = i;

				if(i < 6)
					bestSingleAxis = i;
			}
		}

		REQUIRES(bestCase != 0xffffff);

		if(bestCase < 3)//box2 vert of box1 face
		{
			outContact.push_back(Contact());
			PointFaceContact(box1, box2, toCenter, outContact.back(), bestCase, smallestPen);
			return true;
		}
		else if(bestCase < 6)//box1 vert of box2 face
		{
			outContact.push_back(Contact());
			PointFaceContact(box2, box1, toCenter*-1.0f, outContact.back(), bestCase-3, smallestPen);
			outContact.back().m_contactNormal *= -1;
			return true;
		}
		else//edge on edge action
		{
			bestCase -= 6;

			int axis1Index = bestCase / 3;
			int axis2Index = bestCase % 3;

			Vector3D<Real> axis1 = rotation1.GetAxis(axis1Index);
			Vector3D<Real> axis2 = rotation2.GetAxis(axis2Index);

			Vector3D<Real> axis = axis1.CrossProduct(axis2);
			axis.Normalize();

			if(axis.Dot(toCenter) > 0)
				axis *= -1.0f;

			Vector3D<Real> pointOnEdge1 = box1->GetExtent();
			Vector3D<Real> pointOnEdge2 = box2->GetExtent();
			for(int i = 0; i < 3; ++i)
			{
				if(i == axis1Index)
					pointOnEdge1[i] = 0;
				else if(rotation1.GetAxis(i).Dot(axis) > 0)
					pointOnEdge1[i] = -pointOnEdge1[i];

				if(i == axis2Index)
					pointOnEdge2[i] = 0;
				else if(rotation2.GetAxis(i).Dot(axis) < 0)
					pointOnEdge2[i] = -pointOnEdge2[i];
			}

			pointOnEdge1 = box1->GetTransformation().TransformPosition(pointOnEdge1);
			pointOnEdge2 = box2->GetTransformation().TransformPosition(pointOnEdge2);

			Vector3D<Real> vertex = CalculateContactPoint(
				pointOnEdge1, axis1, box1->GetExtent()[axis1Index],
				pointOnEdge2, axis2, box2->GetExtent()[axis2Index], bestSingleAxis > 2);

			outContact.push_back(Contact());
			auto& newContact = outContact.back();
			newContact.m_penetration = smallestPen;
			newContact.m_contactNormal = axis;
			newContact.m_contactPoint = vertex;

			return true;
		}

		return false;
	}

	bool GeneratePrimitiveContacts(const Primitive* p1, const Primitive* p2, std::vector<Contact>& outContact)
	{
		const Primitive_Sphere* sphere1 = p1->GetType() == CL_Primitive_Sphere ? reinterpret_cast<const Primitive_Sphere*>(p1) : NULL;
		const Primitive_Sphere* sphere2 = p2->GetType() == CL_Primitive_Sphere ? reinterpret_cast<const Primitive_Sphere*>(p2) : NULL;

		const Primitive_Box* box1 = p1->GetType() == CL_Primitive_Box ? reinterpret_cast<const Primitive_Box*>(p1) : NULL;
		const Primitive_Box* box2 = p2->GetType() == CL_Primitive_Box ? reinterpret_cast<const Primitive_Box*>(p2) : NULL;

		const Primitive_Plane* plane1 = p1->GetType() == CL_Primitive_Plane ? reinterpret_cast<const Primitive_Plane*>(p1) : NULL;
		const Primitive_Plane* plane2 = p2->GetType() == CL_Primitive_Plane ? reinterpret_cast<const Primitive_Plane*>(p2) : NULL;
		
		if(sphere1 != NULL && sphere2 != NULL)
			return GenerateContactsSphereSphere(sphere1, sphere2, outContact);
		else if(sphere1 != NULL && plane2 != NULL)
			return GenerateContactsSpherePlane(sphere1, plane2, outContact);
		else if(plane1 != NULL && sphere2 != NULL)
			return GenerateContactsSpherePlane(sphere2, plane1, outContact);
		else if(box1 != NULL && box2 != NULL)
			return GenerateContactsBoxBox(box1, box2, outContact);
		else if(box1 != NULL && plane2 != NULL)
 			return GenerateContactsBoxPlane(box1, plane2, outContact);
		else if(plane1 != NULL && box2 != NULL)
			return GenerateContactsBoxPlane(box2, plane1, outContact);
		else if(sphere1 != NULL && box2 != NULL)
		{
			bool result = GenerateContactsBoxSphere(box2, sphere1, outContact);
			if(result)
				outContact.back().m_contactNormal *= -1;
			return result;
		}
		else if(sphere2 != NULL && box1 != NULL)
		{
			return GenerateContactsBoxSphere(box1, sphere2, outContact);
		}

		return false;
	}
}
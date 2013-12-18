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
 *	@file	CollisionFunctions.h
 *	@brief	Utility function used for generic BoundingVolume collision detection and primitive contact generation.
 *
/****************************************************************************/

#ifndef COLLISIONFUNCTIONS_H_CL
#define COLLISIONFUNCTIONS_H_CL

#include "BoundingVolumes/BoundingVolume.h"
#include "Physics/Primitives/Primitive.h"
#include "Physics/Contact.h"
#include "Math/Vector3D.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	/**
	 * @brief     Detects if b1 and b2 are colliding
	 * @details	  Currently works with AABoundingBox and BoundingSphere, with all combination of params
	 */
	bool BoundingVolumeCollision(const BoundingVolume* b1, const BoundingVolume* b2);


	/**
	 * @brief     Generates contacts between p1 and p2 and adds them to outContact
	 * @details	  Currently works with Primitive_Sphere, Primitive_Box, and Primitive_Plane only
	 */
	bool GeneratePrimitiveContacts(const Primitive* p1, const Primitive* p2, std::vector<Contact>& outContact);
}

#endif
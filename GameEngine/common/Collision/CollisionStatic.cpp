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

#include "CollisionStatic.h"

#include "Physics/Primitives/Primitive_Box.h"
#include "Physics/Primitives/Primitive_Sphere.h"

#include "BoundingVolumes/AABoundingBox.h"
#include "BoundingVolumes/BoundingSphere.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Geometry/Cuboid.h"

namespace CaptainLucha
{
	CollisionStatic::CollisionStatic()
		: m_drawSphere(5.0f, 6)
	{
		m_type = CL_STATIC;
	}

	CollisionStatic::~CollisionStatic()
	{

	}

	void CollisionStatic::Draw(GLProgram& glProgram)
	{
		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->LoadIdentity();

		glProgram.SetUniform("color", m_color);
		glProgram.SetUniform("emissive", 0.0f);

		glProgram.SetUniform("hasDiffuseMap", false);
		glProgram.SetUniform("hasNormalMap", false);
		glProgram.SetUniform("hasSpecularMap", false);
		glProgram.SetUniform("hasEmissiveMap", false);

		if(GetPrimitve()->GetType() == CL_Primitive_Sphere)
		{
			float radius = (float)reinterpret_cast<Primitive_Sphere*>(GetPrimitve())->GetRadius();
			g_MVPMatrix->Scale(radius, radius, radius);
			g_MVPMatrix->Translate(GetPrimitve()->GetTransformation().GetPosition());
			m_drawSphere.Draw(glProgram);
		}
		else if(GetPrimitve()->GetType() == CL_Primitive_Box)
		{
			const Primitive_Box* box = reinterpret_cast<Primitive_Box*>(GetPrimitve());
			Cuboid drawCube(box->GetExtent() * 0.5f);
			g_MVPMatrix->Translate(GetPrimitve()->GetTransformation().GetPosition());
			drawCube.Draw(glProgram);
		}

		g_MVPMatrix->PopMatrix();
	}

	void CollisionStatic::InitCuboid(const Vector3D<Real>& position, const Quaternion& orientation, const Vector3Df& extent)
	{
		Primitive_Box* prim = new Primitive_Box(extent);
		AABoundingBox* aabb = new AABoundingBox(position, extent);

		prim->GetTransformation().SetPosition(position);
		prim->GetTransformation().SetOrientation(orientation);

		InitListener(aabb, prim);
	}
	void CollisionStatic::InitSphere(const Vector3D<Real>& position, float radius)
	{
		Primitive_Sphere* prim = new Primitive_Sphere(radius);
		BoundingSphere* sphere = new BoundingSphere(position, radius);

		prim->GetTransformation().SetPosition(position);

		InitListener(sphere, prim);
	}
}


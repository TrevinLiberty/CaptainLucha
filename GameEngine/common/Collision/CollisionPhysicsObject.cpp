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

#include "CollisionPhysicsObject.h"

#include "CollisionPhysicsObject.h"
#include "CollisionSystem.h"
#include "Math/Quaternion.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Geometry/Sphere.h"
#include "Renderer/Geometry/Cuboid.h"

#include "BoundingVolumes/AABoundingBox.h"
#include "BoundingVolumes/BoundingSphere.h"
#include "Physics/ContactResolver.h"

#include "Physics/Primitives/Primitive_Box.h"
#include "Physics/Primitives/Primitive_Sphere.h"

namespace CaptainLucha
{
	CollisionPhysicsObject::CollisionPhysicsObject()
		: m_isInit(false),
		m_drawSphere(5.0f, 3),
		m_boundingVolume(NULL),
		m_primitive(NULL)
	{
		m_type = CL_PHYSICS_OBJECT;
	}

	CollisionPhysicsObject::~CollisionPhysicsObject()
	{

	}

	void CollisionPhysicsObject::Update(Real DT)
	{
		m_body.Integrate(DT);

		m_boundingVolume->Update(m_body.GetOrientation(), m_body.GetPosition());

		m_primitive->GetTransformation().SetPosition(m_body.GetPosition());
		m_primitive->GetTransformation().SetOrientation(m_body.GetOrientation());

		SetIsActive(m_body.IsAwake());
	}

	void CollisionPhysicsObject::Draw(GLProgram& glProgram)
	{
		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->LoadIdentity();

		glProgram.SetUniform("color", m_color);

		glProgram.SetUniform("hasDiffuseMap", false);
		glProgram.SetUniform("hasNormalMap", false);
		glProgram.SetUniform("hasSpecularMap", false);
        glProgram.SetUniform("hasMaskMap", false);
        glProgram.SetUniform("hasEmissiveMap", false);

		glProgram.SetUniform("emissive", 1.0f);
        glProgram.SetUniform("specularIntensity", 32);

		if(m_primitive->GetType() == CL_Primitive_Sphere)
		{
			float radius = (float)reinterpret_cast<Primitive_Sphere*>(m_primitive)->GetRadius();
			g_MVPMatrix->Scale(radius, radius, radius);
			g_MVPMatrix->LoadMatrix(m_body.GetTransformation());
			m_drawSphere.Draw(glProgram);
		}
		else if(m_primitive->GetType() == CL_Primitive_Box)
		{
			const Primitive_Box* box = reinterpret_cast<Primitive_Box*>(m_primitive);
			Cuboid drawCube(box->GetExtent() * 0.5f);
			g_MVPMatrix->LoadMatrix(m_body.GetTransformation());
			drawCube.Draw(glProgram);
		}

		glProgram.SetUniform("emissive", 0.0f);

		g_MVPMatrix->PopMatrix();
	}

	void CollisionPhysicsObject::InitCuboid(const Vector3D<Real>& extent, Real mass)
	{
		REQUIRES(!m_isInit)

		m_body.SetInertiaTensorCuboid(extent.x, extent.y, extent.z, mass);
		m_body.CalculatedDerivedData();

		m_primitive = new Primitive_Box(extent);
		m_boundingVolume = new AABoundingBox(Vector3D<Real>(), extent);

		InitListener(m_boundingVolume, m_primitive);
		m_isInit = true;
	}

	void CollisionPhysicsObject::InitSphere(Real radius, Real mass)
	{
		REQUIRES(!m_isInit)

		m_body.SetInertiaTensorSphere(radius, mass);
		m_body.CalculatedDerivedData();

		m_primitive = new Primitive_Sphere(radius);
		m_boundingVolume = new BoundingSphere(Vector3D<Real>(), radius);

		InitListener(m_boundingVolume, m_primitive);
		m_isInit = true;
	}

	void CollisionPhysicsObject::SetPosition(const Vector3D<Real>& pos)
	{
		m_body.SetPosition(pos);
		m_boundingVolume->Update(m_body.GetOrientation(), m_body.GetPosition());

		m_primitive->GetTransformation().SetPosition(m_body.GetPosition());
	}

	void CollisionPhysicsObject::Collided(CollisionListener* listener)
	{
		UNUSED(listener)
	}
}
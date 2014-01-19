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
#include "StaticPlane.h"

#include "BoundingVolumes/AABoundingBox.h"
#include "Physics/Primitives/Primitive_Plane.h"
#include "Renderer/RendererUtils.h"
#include "Renderer/Shader/GLProgram.h"

namespace CaptainLucha
{
	StaticPlane::StaticPlane(const Vector3Df& pos, const Vector3Df& normal, const Vector3Df& extent)
		: CollisionStatic(),
		  m_extent(extent)
	{
		SetPosition(pos);
		AABoundingBox* aabb = new AABoundingBox(pos, extent);
		Primitive_Plane* plane = new Primitive_Plane(normal, 0.0f);
		InitListener(aabb, plane);
	}

	StaticPlane::~StaticPlane()
	{

	}

	void StaticPlane::Draw(GLProgram& glProgram)
	{
		SetUtilsColor(m_color);

		glProgram.SetUniform("emissive", 0.0f);
		glProgram.SetUniform("color", m_color);

		glProgram.SetUniform("hasDiffuseMap", false);
		glProgram.SetUniform("hasNormalMap", false);
		glProgram.SetUniform("hasSpecularMap", false);
		glProgram.SetUniform("hasEmissiveMap", false);

		SetGLProgram(&glProgram);
		DrawBegin(CL_QUADS);
		clVertex3(-m_extent.x, 0.0f, -m_extent.z); clNormal3(0.0f, 1.0f, 0.0f);
		clVertex3(-m_extent.x, 0.0f, m_extent.z); clNormal3(0.0f, 1.0f, 0.0f);
		clVertex3(m_extent.x, 0.0f, m_extent.z); clNormal3(0.0f, 1.0f, 0.0f);
		clVertex3(m_extent.x, 0.0f, -m_extent.z); clNormal3(0.0f, 1.0f, 0.0f);
		DrawEnd();
		SetGLProgram(NULL);
	}
}
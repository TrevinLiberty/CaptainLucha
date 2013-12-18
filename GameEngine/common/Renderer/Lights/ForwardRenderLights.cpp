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

#include "ForwardRenderLights.h"
 
#include "../Shader/GLProgram.h"

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	ForwardRenderLights::ForwardRenderLights()
		: m_numLights(0)
	{

	}

	ForwardRenderLights::~ForwardRenderLights()
	{

	}

	void ForwardRenderLights::AddLight(const Vector3Df& pos, const Vector3Df& color)
	{
		if(m_numLights >= MAX_NUM_LIGHTS)
			return;

		m_pos.push_back(pos);
		m_color.push_back(color);
		m_radius.push_back(1.0f);
		m_dir.push_back(Vector3Df());
		m_angleInnerCone.push_back(0.0f);
		m_angleOuterCone.push_back(-1.0f);

		++m_numLights;
	}

	void ForwardRenderLights::SetLightPoint(int index, float radius)
	{
		if(IsIndexValid(index))
		{
			m_radius[index] = radius;
			m_dir[index] = Vector3Df();
			m_angleInnerCone[index] = 0.0f;
			m_angleOuterCone[index] = 0.0f;
		}
	}

	void ForwardRenderLights::SetLightSpotLight(int index, float radius, const Vector3Df& dir, float outAngle, float inAngle)
	{
		if(IsIndexValid(index))
		{
			m_radius[index] = radius;
			m_dir[index] = dir;
			m_angleInnerCone[index] = cos(DegreesToRadians(inAngle));
			m_angleOuterCone[index] = cos(DegreesToRadians(outAngle));
		}
	}

	void ForwardRenderLights::ApplyLights(GLProgram& glProgram)
	{
		const int PROGRAM_ID = glProgram.GetProgramID();

		if(m_numLights == 0)
			return;

		const int LIGHT_POS_LOC = glGetUniformLocation(PROGRAM_ID, "lightPositions");
		const int LIGHT_COLOR_LOC = glGetUniformLocation(PROGRAM_ID, "lightColors");
		const int LIGHT_DIR_LOC = glGetUniformLocation(PROGRAM_ID, "lightDirs");
		const int LIGHT_RADIUS_LOC = glGetUniformLocation(PROGRAM_ID, "lightradius");
		const int LIGHT_INNER_LOC = glGetUniformLocation(PROGRAM_ID, "lightInnerCone");
		const int LIGHT_OUTER_LOC = glGetUniformLocation(PROGRAM_ID, "lightOuterCone");

		glProgram.SetUniform("numLights", m_numLights);

		if (LIGHT_POS_LOC >= 0 && !m_pos.empty())
		{
			glUniform3fv(LIGHT_POS_LOC, m_numLights, &m_pos[0].x);
		}
		if (LIGHT_COLOR_LOC >= 0 && !m_color.empty())
		{
			glUniform3fv(LIGHT_COLOR_LOC, m_numLights, &m_color[0].x);
		}
		if (LIGHT_DIR_LOC >= 0 && !m_dir.empty())
		{
			glUniform3fv(LIGHT_DIR_LOC, m_numLights, &m_dir[0].x);
		}
		if (LIGHT_RADIUS_LOC >= 0 && !m_radius.empty())
		{
			glUniform1fv(LIGHT_RADIUS_LOC, m_numLights, &m_radius[0]);
		}
		if (LIGHT_INNER_LOC >= 0 && !m_angleInnerCone.empty())
		{
			glUniform1fv(LIGHT_INNER_LOC, m_numLights, &m_angleInnerCone[0]);
		}
		if (LIGHT_OUTER_LOC >= 0 && !m_angleOuterCone.empty())
		{
			glUniform1fv(LIGHT_OUTER_LOC, m_numLights, &m_angleOuterCone[0]);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////	
	bool ForwardRenderLights::IsIndexValid( int index ) const
	{
		return index < m_numLights;
	}

}
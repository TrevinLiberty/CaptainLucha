#include "Lights.h"
 
#include "Shader/GLProgram.h"

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	Lights::Lights()
		: m_numLights(0)
	{

	}

	Lights::~Lights()
	{

	}

	void Lights::AddLight(const Vector3Df& pos, const Vector3Df& color)
	{
		if(m_numLights >= MAX_NUM_LIGHTS)
			return;

		m_pos.push_back(pos);
		m_color.push_back(color);
		m_radius.push_back(1.0f);
		m_dir.push_back(Vector3Df());
		m_angleInnerCone.push_back(0.0f);
		m_angleOuterCone.push_back(-1.0f);
		m_startAndEndAtten.push_back(Vector2Df(1.0f, 1.0f));

		++m_numLights;
	}

	void Lights::SetLightPoint(int index, float radius)
	{
		if(IsIndexValid(index))
		{
			m_radius[index] = radius;
			m_dir[index] = Vector3Df();
			m_angleInnerCone[index] = 0.0f;
			m_angleOuterCone[index] = 0.0f;
		}
	}

	void Lights::SetLightSpotLight(int index, float radius, const Vector3Df& dir, float outAngle, float inAngle)
	{
		if(IsIndexValid(index))
		{
			m_radius[index] = radius;
			m_dir[index] = dir;
			m_angleInnerCone[index] = cos(DegreesToRadians(inAngle));
			m_angleOuterCone[index] = cos(DegreesToRadians(outAngle));
		}
	}

	void Lights::SetLightAttenStartAndEnd(int index, float start, float end)
	{
		if(IsIndexValid(index))
		{
			m_startAndEndAtten[index].x_ = start;
			m_startAndEndAtten[index].y_ = end;
		}
	}

	void Lights::ApplyLights(GLProgram& glProgram)
	{
		const int PROGRAM_ID = glProgram.GetProgramID();

		const int LIGHT_POS_LOC = glGetUniformLocation(PROGRAM_ID, "lightPositions");
		const int LIGHT_COLOR_LOC = glGetUniformLocation(PROGRAM_ID, "lightColors");
		const int LIGHT_DIR_LOC = glGetUniformLocation(PROGRAM_ID, "lightDirs");
		const int LIGHT_RADIUS_LOC = glGetUniformLocation(PROGRAM_ID, "lightradius");
		const int LIGHT_INNER_LOC = glGetUniformLocation(PROGRAM_ID, "lightInnerCone");
		const int LIGHT_OUTER_LOC = glGetUniformLocation(PROGRAM_ID, "lightOuterCone");
		const int LIGHT_STARTENDATTEN_LOC = glGetUniformLocation(PROGRAM_ID, "lightAttenStartEnd");

		glProgram.SetUniform("numLights", m_numLights);

		if (LIGHT_POS_LOC >= 0 && !m_pos.empty())
		{
			glUniform3fv(LIGHT_POS_LOC, m_numLights, &m_pos[0].x_);
		}
		if (LIGHT_COLOR_LOC >= 0 && !m_color.empty())
		{
			glUniform3fv(LIGHT_COLOR_LOC, m_numLights, &m_color[0].x_);
		}
		if (LIGHT_DIR_LOC >= 0 && !m_dir.empty())
		{
			glUniform3fv(LIGHT_DIR_LOC, m_numLights, &m_dir[0].x_);
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
		if(LIGHT_STARTENDATTEN_LOC >= 0 && !startAndEndAtten_.empty())
		{
			glUniform2fv(LIGHT_STARTENDATTEN_LOC, numLights_, &startAndEndAtten_[0]);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////	
	bool Lights::IsIndexValid( int index ) const
	{
		return index < m_numLights;
	}

}
#include "Quad.h"

namespace CaptainLucha
{
	Quad::Quad(const Vector3Df& v0, const Vector3Df& v1, const Vector3Df& v2, const Vector3Df& v3)
	{
		std::vector<TangentSpaceVertex> verts;
		verts.push_back(TangentSpaceVertex())
	}

	Quad::~Quad()
	{

	}

	void Quad::Draw(GLProgram& glProgram)
	{
		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->Translate(m_pos);
		m_vbo->Draw(glProgram);
		g_MVPMatrix->PopMatrix();
	}

}
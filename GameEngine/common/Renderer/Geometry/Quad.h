///////////////////////////////////////////////////////////////////////
//
//	Author: Trevin Liberty
//	Data:   7/9/2013
//	Description:
//		
//
///////////////////////////////////////////////////////////////////////

#ifndef PLANE_H_CL
#define PLANE_H_CL

#include "Math/Vector3D.h"
#include "Renderer/VertexBufferObject.h"

namespace CaptainLucha
{
	class GLProgram;

	class Quad
	{
	public:
		Quad(const Vector3Df& v0, const Vector3Df& v1, const Vector3Df& v2, const Vector3Df& v3);
		~Quad();

		void Draw(GLProgram& glProgram);

		const Vector3Df& GetPos() const {return m_pos;}
		void SetPos(const Vector3Df& val) {m_pos = val;}

	protected:

	private:
		Vector3Df m_pos;

		VertexBufferObject* m_vbo;
	};
}

#endif
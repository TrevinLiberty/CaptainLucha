///////////////////////////////////////////////////////////////////////
//
//	Author: Trevin Liberty
//	Data:   8/2/2013
//	Description:
//		
//
///////////////////////////////////////////////////////////////////////

#ifndef ORIENTEDBOUNDINGBOX_H_CL
#define ORIENTEDBOUNDINGBOX_H_CL

#include "Math/Vector3D.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class OrientedBoundingBox
	{
	public:
		OrientedBoundingBox();
		~OrientedBoundingBox();

	protected:

	private:
		Vector3Df m_center;
		Vector3Df m_localAxis[3];
		Vector3Df m_extent;
	};
}

#endif
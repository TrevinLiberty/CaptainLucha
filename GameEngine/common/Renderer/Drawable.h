///////////////////////////////////////////////////////////////////////
//
//	Author: Trevin Liberty
//	Data:   10/26/2013
//	Description:
//		
//
///////////////////////////////////////////////////////////////////////

#ifndef DRAWABLE_H_CL
#define DRAWABLE_H_CL

namespace CaptainLucha
{
	class Drawable
	{
	public:
		Drawable() {};
		virtual ~Drawable() {};

		virtual void Draw(GLProgram& glProgram) {UNUSED(glProgram)};
	};
}

#endif
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
 *	@file	ForwardRenderLights.h
 *	@brief	
 *
/****************************************************************************/

#ifndef LIGHT_H_CL
#define LIGHT_H_CL

#include "Math/Vector2D.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "Utils/UtilMacros.h"

#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class GLProgram;

	enum LightFallOffType
	{
		CL_NONE,
		CL_LINEAR,
		CL_INVERSE,
		CL_INVERSE_SQRD
	};

	class ForwardRenderLights
	{
	public:
		static const int MAX_NUM_LIGHTS = 14;

		ForwardRenderLights();
		~ForwardRenderLights();

		void AddLight(const Vector3Df& pos, const Vector3Df& color);

		void SetLightPoint(int index, float radius);
		void SetLightSpotLight(int index, float radius, const Vector3Df& dir, float outDegrees, float inDegrees);

		void SetLightPos(int index, const Vector3Df& pos)			 {AddToVector(m_pos, index, pos);}
		void SetLightColor(int index, const Vector3Df& color)		 {AddToVector(m_color, index, color);}
		void SetLightDir(int index, const Vector3Df& dir)			 {AddToVector(m_dir, index, dir);}
		void SetLightRadius(int index, float radius)				 {AddToVector(m_radius, index, radius);}
		void SetLightAngleInnerCone(int index, float InnerConeAngle) {AddToVector(m_angleInnerCone, index, (float)cos(DegreesToRadians(InnerConeAngle)));}
		void SetLightAngleOuterCone(int index, float OuterConeAngle) {AddToVector(m_angleOuterCone, index, (float)cos(DegreesToRadians(OuterConeAngle)));}

		const Vector3Df& GetLightPos(int index) {return m_pos[index];}
		const Vector3Df& GetLightDir(int index) {return m_dir[index];}
		const Vector3Df& GetLightColor(int index) {return m_color[index];}

	protected:
		bool IsIndexValid(int index) const;

		template<class T>
		void AddToVector(std::vector<T>& vect, int index, T val)
		{
			if(IsIndexValid(index))
				vect[index] = val;
		}

		void ApplyLights(GLProgram& glProgram);

	private:
		std::vector<Vector3Df> m_pos;
		std::vector<Vector3Df> m_color;
		std::vector<Vector3Df> m_dir;

		std::vector<float> m_radius;
		std::vector<float> m_angleInnerCone;
		std::vector<float> m_angleOuterCone;

		int m_numLights;

		friend class GLProgram;

		PREVENT_COPYING(ForwardRenderLights)
	};
}

#endif
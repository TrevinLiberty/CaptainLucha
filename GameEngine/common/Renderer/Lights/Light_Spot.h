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
 *	@file	Light_Spot.h
 *	@brief	
 *
/****************************************************************************/

#ifndef LIGHT_SPOT_H_CL
#define LIGHT_SPOT_H_CL

#include "Light_Directional.h"

#include "Utils/Utils.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class Light_Spot : public Light
	{
	public:
		Light_Spot() : Light(CL_SPOT_LIGHT) {};
		~Light_Spot() {};

		void SetLookAt(const Vector3Df& lookAt);

		const float GetInnerConeAngle() const {return m_innerConeAngle;}
		void SetInnerConeAngle(const float& degrees) {m_innerConeAngle = DegreesToRadians(degrees);}

		const float GetOuterConeAngle() const {return m_outerConeAngle;}
		void SetOuterConeAngle(const float& degrees) {m_outerConeAngle = DegreesToRadians(degrees);}

		const Vector3Df& GetLightDir() const {return m_lightDir;}
		void SetLightDir(const Vector3Df& val) {m_lightDir = val;}

	protected:
		float m_innerConeAngle;
		float m_outerConeAngle;

		Vector3Df m_lightDir;
	};
}

#endif
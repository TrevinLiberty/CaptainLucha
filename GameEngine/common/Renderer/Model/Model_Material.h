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
 *	@file	Model_Material.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MODEL_MATERIAL_H_CL
#define MODEL_MATERIAL_H_CL

#include "Renderer/Color.h"

namespace CaptainLucha
{
	class GLProgram;
	class GLTexture;

	class Model_Material
	{
	public:
		Model_Material();
		~Model_Material();

		void ApplyMaterial(GLProgram& glProgram);

		void Diffuse(GLTexture* val) {m_diffuse = val;}
		void Normal(GLTexture* val) {m_normal = val;}
		void Specular(GLTexture* val) {m_specular = val;}
		void Emissive(GLTexture* val) {m_emissive = val;}
		void Mask(GLTexture* val) {m_mask = val;}

		void DiffuseColor(const Color& val) {m_diffuseColor = val;}
		void SpecularColor(const Color& val) {m_specularColor = val;}
		void EmissiveColor(const Color& val) {m_emissiveColor = val;}

		void Opacity(const float& val) {m_opacity = val;}
		void SpecIntensity(const float& val) {m_specIntensity = val;}

	protected:

	private:
		GLTexture* m_diffuse;
		GLTexture* m_normal;
		GLTexture* m_specular;
		GLTexture* m_emissive;
		GLTexture* m_mask;

		Color m_diffuseColor;
		Color m_specularColor;
		Color m_emissiveColor;

		float m_opacity;
		float m_specIntensity;
	};
}

#endif
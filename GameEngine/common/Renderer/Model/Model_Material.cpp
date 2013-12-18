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

#include "Model_Material.h"

#include "Renderer/Shader/GLProgram.h"
#include "Renderer/Texture/GLTexture.h"

namespace CaptainLucha
{
	Model_Material::Model_Material()
		: m_diffuse(NULL),
		  m_emissive(NULL),
		  m_normal(NULL),
		  m_specular(NULL)
	{

	}

	Model_Material::~Model_Material()
	{

	}

	void Model_Material::ApplyMaterial(GLProgram& glProgram)
	{
		glProgram.SetUniform("emissive", 0.0f);

		glProgram.SetUnifromTexture("diffuseMap", m_diffuse);
		glProgram.SetUnifromTexture("normalMap", m_normal);
		glProgram.SetUnifromTexture("specularMap", m_specular);
		glProgram.SetUnifromTexture("emissiveMap", m_emissive);
		glProgram.SetUnifromTexture("maskMap", m_mask);

		glProgram.SetUniform("hasDiffuseMap", m_diffuse != NULL);
		glProgram.SetUniform("hasNormalMap", m_normal != NULL);
		glProgram.SetUniform("hasSpecularMap", m_specular != NULL);
		glProgram.SetUniform("hasEmissiveMap", m_emissive != NULL);
		glProgram.SetUniform("hasMaskMap", m_mask != NULL);

		glProgram.SetUniform("color", m_diffuseColor);
		glProgram.SetUniform("specularColor", m_specularColor);
		glProgram.SetUniform("emissiveColor", m_emissiveColor);

		glProgram.SetUniform("opacity", m_opacity);
		glProgram.SetUniform("specularIntensity", m_specIntensity);
	}

}
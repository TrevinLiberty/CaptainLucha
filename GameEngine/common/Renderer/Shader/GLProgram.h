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
 *	@file	GLProgram.h
 *	@brief	
 *
/****************************************************************************/

#ifndef GLPROGRAM_H_CL
#define GLPROGRAM_H_CL

#include "Renderer/Texture/GLTexture.h"
#include "Renderer/Lights/ForwardRenderLights.h"

#include <glew.h>

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	struct UniformBase;

	typedef std::map<unsigned int, UniformBase*>::iterator UnifromIterator;

	class GLProgram
	{
	public:
		GLProgram(const char* vertPath, const char* fragPath);
		~GLProgram();

		int GetAttributeLocation(const std::string& name);
		int GetProgramID() const {return m_programID;}

		template <typename T>
		void SetUniform(const std::string& name, const T& val);

		void SetUnifromTexture(const std::string& name, GLTexture* glTexture);

		void SetModelViewProjection();

		void UseProgram();

		void ClearTextureUnits();

		ForwardRenderLights& GetLights() {return m_lights;}

	protected:
		void CreateProgram(const char* vertPath, const char* fragPath);

		template <typename T>
		UniformBase* CreateNewUniform(const T& val);

		void OutputShaderError(int programID, const char* fileName);
		void OutputProgramError(int programID, const char* vert, const char* frag);

	private:
		unsigned int m_programID;

		int m_nextValidTextureUnit;

		const char* m_vertShaderName;
		const char* m_fragShaderName;

		ForwardRenderLights m_lights;

		std::map<unsigned int, UniformBase*> m_uniforms;
		std::map<unsigned int, unsigned int> m_attributeLocs;
	};

	template <typename T>
	void GLProgram::SetUniform(const std::string& name, const T& val)
	{
		unsigned int hash = ByteHash(name.c_str(), name.size());
		auto it = m_uniforms.find(hash);

		if(it != m_uniforms.end())
		{
			int cachedUniformLoc = it->second->m_uniformLoc;
			delete it->second;
			it->second = CreateNewUniform(val);
			it->second->m_uniformLoc = cachedUniformLoc;
		}
		else
		{
			UniformBase*& newUniform = m_uniforms[hash];
			newUniform = CreateNewUniform(val);
			newUniform->m_uniformLoc = glGetUniformLocation(m_programID, name.c_str());
		}
	}

	struct UniformBase
	{
		int m_uniformLoc;

		virtual void SetUniform() = 0;
	};

	template <typename T>
	struct Uniform : UniformBase
	{
		int m_size;
		T m_values[4];

		void SetUniform();
	};
}

#endif
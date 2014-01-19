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
#include "Time/ProfilingSection.h"
#include "Renderer/RendererUtils.h"

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

		void SetUnifromTexture(const std::string& name, GLTexture* glTexture, bool isCubeMap = false);

		void SetModelViewProjection();

		void UseProgram();

		void ClearTextureUnits();

		ForwardRenderLights& GetLights() {return m_lights;}

        void AddTesselationShaders(const char* ctrlPath, const char* evalPath);
        void AddGeometryShader(const char* geoPath);

	protected:
   		template <typename T>
		UniformBase* CreateNewUniform(const T& val)
        {
            return new Uniform<T>(val);
        }

		void OutputShaderError(int programID, const char* fileName);
		void OutputProgramError(int programID, const char* vert, const char* frag);

        void CompileAndAttachShader(const char* path, GLenum shaderType);
        void LinkProgram();

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
        ProfilingSection p("Set Uniform");
        std::map<unsigned int, UniformBase*>::iterator it;
        unsigned int hash;
        {
            ProfilingSection p("find Uniform");
            {
                ProfilingSection p("byte hash");
                hash = ByteHash(name.c_str(), name.size());
            }
            it = m_uniforms.find(hash);
        }

		if(it != m_uniforms.end())
		{
            ProfilingSection p("Change Uniform");
            Uniform<T>* uniform = dynamic_cast<Uniform<T>* >(it->second);
            if(uniform)
            {
                ProfilingSection p("Update");
                if(!uniform->Equals(val))
                {
                    uniform->m_value = val;
                }
            }
            else
            {
                ProfilingSection p("Replace");
                int cachedUniformLoc = it->second->m_uniformLoc;
                delete it->second;
                it->second = CreateNewUniform(val);
                it->second->m_uniformLoc = cachedUniformLoc;
            }
		}
		else
		{
            ProfilingSection p("New Uniform");
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
        Uniform(const T& val)
            : m_value(val) {}

		T m_value;

		void SetUniform();

        bool Equals(const T& val)
        {
            return m_value == val;
        }
	};

    struct UniformTexture : Uniform<GLTexture*>
    {
        UniformTexture(GLTexture* val)
            : Uniform(val) {};

        int m_textureUnit;
        bool m_isCubeMap;
    };
}

#endif
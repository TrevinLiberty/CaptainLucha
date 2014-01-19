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

#include "GLProgram.h"
#include "Utils/CLFileImporter.h"

#include "Math/Matrix4D.h"
#include "Math/Vector2D.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Texture/GLTexture.h"
#include "Time/ProfilingSection.h"

#include <glew.h>
#include <glfw3.h>

//#define DEBUG_GL_PROGRAM

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	GLProgram::GLProgram(const char* vertPath, const char* fragPath)
		: m_nextValidTextureUnit(0)
	{
        m_programID = glCreateProgram();
        glBindFragDataLocation(m_programID, 0, "FragColor");

        CompileAndAttachShader(vertPath, GL_VERTEX_SHADER);
        CompileAndAttachShader(fragPath, GL_FRAGMENT_SHADER);
        LinkProgram();
	}

	GLProgram::~GLProgram()
	{
		glDeleteProgram(m_programID);

		for(auto it = m_uniforms.begin(); it != m_uniforms.end(); ++it)
			delete it->second;
	}

	int GLProgram::GetAttributeLocation(const std::string& name)
	{
		unsigned int hashVal = ByteHash(name.c_str(), name.size());
		auto it = m_attributeLocs.find(hashVal);
		int result = -1;

		if(it != m_attributeLocs.end())
		{
			result = it->second;
		}
		else
		{
			result = glGetAttribLocation(m_programID, name.c_str());

			if(result < 0)
			{
#ifdef DEBUG_GL_PROGRAM 
				traceWithFile("ERROR: Unable to find Attribute " << name); 
#endif
			}
			else
				m_attributeLocs[hashVal] = result;
		}

		return result;
	}

	void GLProgram::SetUnifromTexture(const std::string& name, GLTexture* glTexture, bool isCubeMap)
	{
		unsigned int hash = ByteHash(name.c_str(), name.size());
		UniformBase*& uniform = m_uniforms[hash];

		if(uniform != NULL)
		{
			UniformTexture* textureUniform = dynamic_cast<UniformTexture*>(uniform);

			if(!textureUniform)
			{
				std::stringstream ss;
				ss << "ERROR: SetUniformTexture(...) /nError Setting texture to slot: " << name << "/nUniform isn't a texture";
				MessageBox(NULL, "UNSUPPORTED TEXTURE FORMAT", ss.str().c_str(), MB_OK | MB_ICONERROR);
			}
			else
				textureUniform->m_value = glTexture;
		}
		else
		{
			UniformTexture* newUniform = new UniformTexture(glTexture);
			newUniform->m_uniformLoc = glGetUniformLocation(m_programID, name.c_str());
			newUniform->m_textureUnit = m_nextValidTextureUnit;
            newUniform->m_isCubeMap = isCubeMap;
			++m_nextValidTextureUnit;
			uniform = newUniform;
		}
	}

	//todo Remove model, view, projection uniforms when shaders are updated
	void GLProgram::SetModelViewProjection()
	{
		SetUniform("model", g_MVPMatrix->GetModelMatrix());
		SetUniform("view", g_MVPMatrix->GetViewMatrix());
		SetUniform("projection", g_MVPMatrix->GetProjectionMatrix());
		SetUniform("modelViewProj", g_MVPMatrix->GetModelViewProjectionMatrix());
	}

	void GLProgram::UseProgram()
	{
		glUseProgram(m_programID);

		for(auto it = m_uniforms.begin(); it != m_uniforms.end(); ++it)
		{
			if(it->second->m_uniformLoc != -1)
				it->second->SetUniform();
       	}

		m_lights.ApplyLights(*this);
	}

	void GLProgram::ClearTextureUnits()
	{
		glEnable(GL_TEXTURE_2D);
		for(int i = 0; i < m_nextValidTextureUnit; ++i)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_2D, 0);
			glActiveTexture(GL_TEXTURE0);
		}
		glDisable(GL_TEXTURE_2D);
	}

    void GLProgram::AddTesselationShaders(const char* ctrlPath, const char* evalPath)
    {
        GLint MaxPatchVertices = 0;
        glGetIntegerv(GL_MAX_PATCH_VERTICES, &MaxPatchVertices);
        printf("Max supported patch vertices %d\n", MaxPatchVertices);	
        glPatchParameteri(GL_PATCH_VERTICES, 3);

        CompileAndAttachShader(ctrlPath, GL_TESS_CONTROL_SHADER);
        CompileAndAttachShader(evalPath, GL_TESS_EVALUATION_SHADER);
        LinkProgram();
    }

    void GLProgram::AddGeometryShader(const char* geoPath)
    {
        CompileAndAttachShader(geoPath, GL_GEOMETRY_SHADER);
        LinkProgram();
    }

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void GLProgram::OutputShaderError(int programID, const char* fileName)
	{
		int length;
		glGetShaderiv(programID, GL_INFO_LOG_LENGTH, &length);

		char* log = new char[length];
		glGetShaderInfoLog(programID, length, &length, log);

		std::string dirPath = GetCurrentDirectoryPath_w() + "\\" + fileName + "(";
		std::stringstream logSS(log);

		trace("\nShader Compile Error:" << fileName)

            while(!logSS.eof())
            {
                std::string line;
                getline(logSS, line);
                trace(dirPath + (line.c_str() + 2))
            }

            std::cout << log << std::endl;

            MessageBox(NULL, log, "Shader ERROR!", MB_ICONERROR);
	}

	void GLProgram::OutputProgramError(int programID, const char* vert, const char* frag)
	{
		int length;
		glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &length);

		char* log = new char[length];
		glGetProgramInfoLog(programID, length, &length, log);

		std::string dirPath = GetCurrentDirectoryPath_w() + "\\" + vert + "(";

		std::stringstream logSS(log);

		traceWithFile("\nProgram Linking Error\n");

		while(!logSS.eof())
		{
			std::string line;
			getline(logSS, line);

			if(line.size() > 0 && line.front() == 'F')
			{
				dirPath = GetCurrentDirectoryPath_w() + "\\" + frag + "(";
				trace(line)
				continue;
			}

			if(line.size() > 0 && (line.front() != '-' && line.front() != 'V'))
				trace(dirPath << (line.c_str() + 2))
			else
				trace(line)
		}

		std::cout << log << std::endl;
		MessageBox(NULL, log, "Shader Program ERROR!", MB_OK | MB_ICONERROR);
	}

    void GLProgram::CompileAndAttachShader(const char* path, GLenum shaderType)
    {
        CLFileImporter fileImporter;
        CLFile* file = fileImporter.LoadFile(path);

        if(file->m_buffer == NULL || file->m_bufLength <= 0)
        {
            std::stringstream ss;
            ss << "ERROR: Unable to open shader file " << path;
            MessageBox(NULL, "UNSUPPORTED TEXTURE FORMAT", ss.str().c_str(), MB_OK | MB_ICONERROR);
        }

        GLuint shader = glCreateShader(shaderType);
        glShaderSource(shader, 1, (const char**)&file->m_buffer, &file->m_bufLength);
        glCompileShader(shader);

        int success = static_cast<int>(false);
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success)
            OutputShaderError(shader, path);

        glAttachShader(m_programID, shader);

        glDeleteShader(shader);
        delete file;
    }

    void GLProgram::LinkProgram()
    {
        int success = static_cast<int>(false);
        glLinkProgram(m_programID);
        glGetProgramiv(m_programID, GL_LINK_STATUS, &success);
        if(!success)
            OutputProgramError(m_programID, "", "");
    }

	//////////////////////////////////////////////////////////////////////////
	//	Uniform Base
	//////////////////////////////////////////////////////////////////////////
	void Uniform<bool>::SetUniform()
	{
		glUniform1i(m_uniformLoc, m_value);
	}

	void Uniform<int>::SetUniform()
	{
		glUniform1i(m_uniformLoc, m_value);
	}

	void Uniform<unsigned int>::SetUniform()
	{
		glUniform1ui(m_uniformLoc, m_value);
	}

	void Uniform<float>::SetUniform()
	{
		glUniform1f(m_uniformLoc, m_value);
	}

	void Uniform<double>::SetUniform()
	{
		glUniform1f(m_uniformLoc, (float)m_value);
	}

    void Uniform<Vector2Df>::SetUniform()
    {
        glUniform2f(m_uniformLoc, m_value.x, m_value.y);
    }

    void Uniform<Vector3Df>::SetUniform()
    {
        glUniform3f(m_uniformLoc, m_value.x, m_value.y, m_value.z);
    }

    void Uniform<Vector4Df>::SetUniform()
    {
        glUniform4f(m_uniformLoc, m_value.x, m_value.y, m_value.z, m_value.w);
    }

    void Uniform<Color>::SetUniform()
    {
        glUniform4f(m_uniformLoc, m_value.r, m_value.g, m_value.b, m_value.a);
    }

	void Uniform<Matrix4Df>::SetUniform()
	{
		glUniformMatrix4fv(m_uniformLoc, 1, GL_FALSE, m_value.Data());
	}

	void Uniform<GLTexture*>::SetUniform()
	{
		if(m_value)
		{
            const int TEXTURE_UNIT = reinterpret_cast<UniformTexture*>(this)->m_textureUnit;
            const bool IS_CUBE_MAP = reinterpret_cast<UniformTexture*>(this)->m_isCubeMap;

			glEnable(GL_TEXTURE_2D);
			glUniform1i(m_uniformLoc, TEXTURE_UNIT);
			glActiveTexture(GL_TEXTURE0 + TEXTURE_UNIT);

            if(IS_CUBE_MAP)
                glBindTexture(GL_TEXTURE_CUBE_MAP, m_value->GetTextureID());
            else
			    glBindTexture(GL_TEXTURE_2D, m_value->GetTextureID());


			glActiveTexture(GL_TEXTURE0);
			glDisable(GL_TEXTURE_2D);
		}
	}
}
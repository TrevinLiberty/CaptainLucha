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
		CreateProgram(vertPath, fragPath);
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

	void GLProgram::SetUnifromTexture(const std::string& name, GLTexture* glTexture)
	{
		unsigned int hash = ByteHash(name.c_str(), name.size());
		UniformBase*& uniform = m_uniforms[hash];

		if(uniform != NULL)
		{
			Uniform<GLTexture*>* textureUniform = dynamic_cast<Uniform<GLTexture*>*>(uniform);

			if(!textureUniform)
			{
				std::stringstream ss;
				ss << "ERROR: SetUniformTexture(...) /nError Setting texture to slot: " << name << "/nUniform isn't a texture";
				MessageBox(NULL, "UNSUPPORTED TEXTURE FORMAT", ss.str().c_str(), MB_OK | MB_ICONERROR);
			}
			else
				textureUniform->m_values[0] = glTexture;
		}
		else
		{
			Uniform<GLTexture*>* newUniform = new Uniform<GLTexture*>();
			newUniform->m_uniformLoc = glGetUniformLocation(m_programID, name.c_str());
			newUniform->m_size = m_nextValidTextureUnit;
			++m_nextValidTextureUnit;
			newUniform->m_values[0] = glTexture;
			uniform = newUniform;
		}
	}

	void GLProgram::SetModelViewProjection()
	{
		SetUniform("model", g_MVPMatrix->GetModelMatrix());
		SetUniform("view", g_MVPMatrix->GetViewMatrix());
		SetUniform("projection", g_MVPMatrix->GetProjectionMatrix());
	}

	void GLProgram::UseProgram()
	{
		glUseProgram(m_programID);
		int error = glGetError();
		if(error)
		{
			error += 10;
		}

		for(auto it = m_uniforms.begin(); it != m_uniforms.end(); ++it)
		{
			error = glGetError();
			if(error)
			{
				error += 10;
			}

			if(it->second->m_uniformLoc != -1)
				it->second->SetUniform();

			error = glGetError();
			if(error)
			{
				error += 10;
			}
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

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void GLProgram::CreateProgram(const char* vertPath, const char* fragPath)
	{
		CLFileImporter fileImporter;
		CLFile* vertexFile = fileImporter.LoadFile(vertPath);
		CLFile* fragFile = fileImporter.LoadFile(fragPath);

		if(vertexFile->m_buffer == NULL || vertexFile->m_bufLength <= 0)
		{
			std::stringstream ss;
			ss << "ERROR: Unable to open shader file " << vertPath;
			MessageBox(NULL, "UNSUPPORTED TEXTURE FORMAT", ss.str().c_str(), MB_OK | MB_ICONERROR);
		}

		if(fragFile->m_buffer == NULL || fragFile->m_bufLength <= 0)
		{
			std::stringstream ss;
			ss << "ERROR: Unable to open shader file " << fragPath;
			MessageBox(NULL, "UNSUPPORTED TEXTURE FORMAT", ss.str().c_str(), MB_OK | MB_ICONERROR);
		}

		GLuint shV = glCreateShader(GL_VERTEX_SHADER);
		GLuint shF = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(shV, 1, (const char**)&vertexFile->m_buffer, &vertexFile->m_bufLength);
		glShaderSource(shF, 1, (const char**)&fragFile->m_buffer, &fragFile->m_bufLength);
		glCompileShader(shV);
		glCompileShader(shF);

		int success = static_cast<int>(false);
		glGetShaderiv(shV, GL_COMPILE_STATUS, &success);
		if(!success)
			OutputShaderError(shV, vertPath);
		success = static_cast<int>(false);
		glGetShaderiv(shF, GL_COMPILE_STATUS, &success);
		if(!success)
			OutputShaderError(shV, fragPath);

		m_programID = glCreateProgram();
		glAttachShader(m_programID, shV);
		glAttachShader(m_programID, shF);
		glLinkProgram(m_programID);
		glGetProgramiv(m_programID, GL_LINK_STATUS, &success);
		if(!success)
			OutputProgramError(m_programID, vertPath, fragPath);

		m_vertShaderName = vertPath;
		m_fragShaderName = fragPath;

		glDeleteShader(shV);
		glDeleteShader(shF);
		glBindFragDataLocation(m_programID, 0, "FragColor");

		delete vertexFile;
		delete fragFile;
	}

	template <>
	UniformBase* GLProgram::CreateNewUniform<bool>(const bool& val)
	{
		Uniform<bool>* result = new Uniform<bool>();
		result->m_size = 1;
		result->m_values[0] = val;
		return result;
	}

	template <>
	UniformBase* GLProgram::CreateNewUniform<int>(const int& val)
	{
		Uniform<int>* result = new Uniform<int>();
		result->m_size = 1;
		result->m_values[0] = val;
		return result;
	}

	template <>
	UniformBase* GLProgram::CreateNewUniform<unsigned int>(const unsigned int& val)
	{
		Uniform<unsigned int>* result = new Uniform<unsigned int>();
		result->m_size = 1;
		result->m_values[0] = val;
		return result;
	}

	template<>
	UniformBase* GLProgram::CreateNewUniform<float>(const float& val)
	{
		Uniform<float>* result = new Uniform<float>();
		result->m_size = 1;
		result->m_values[0] = val;
		return result;
	}

	template<>
	UniformBase* GLProgram::CreateNewUniform<double>(const double& val)
	{
		Uniform<double>* result = new Uniform<double>();
		result->m_size = 1;
		result->m_values[0] = val;
		return result;
	}

	template<>
	UniformBase* GLProgram::CreateNewUniform<Vector2Df>(const Vector2Df& val)
	{
		Uniform<float>* result = new Uniform<float>();
		result->m_size = 2;
		result->m_values[0] = val.x;
		result->m_values[1] = val.y;
		return result;
	}

	template<>
	UniformBase* GLProgram::CreateNewUniform<Vector3Df>(const Vector3Df& val)
	{
		Uniform<float>* result = new Uniform<float>();
		result->m_size = 3;
		result->m_values[0] = val.x;
		result->m_values[1] = val.y;
		result->m_values[2] = val.z;
		return result;
	}

	template<>
	UniformBase* GLProgram::CreateNewUniform<Vector4Df>(const Vector4Df& val)
	{
		Uniform<float>* result = new Uniform<float>();
		result->m_size = 4;
		result->m_values[0] = val.x;
		result->m_values[1] = val.y;
		result->m_values[2] = val.z;
		result->m_values[3] = val.w;
		return result;
	}

	template<>
	UniformBase* GLProgram::CreateNewUniform<Color>(const Color& val)
	{
		Uniform<float>* result = new Uniform<float>();
		result->m_size = 4;
		result->m_values[0] = val.r;
		result->m_values[1] = val.g;
		result->m_values[2] = val.b;
		result->m_values[3] = val.a;
		return result;
	}

	template<>
	UniformBase* GLProgram::CreateNewUniform<Matrix4Df>(const Matrix4Df& val)
	{
		Uniform<Matrix4Df>* result = new Uniform<Matrix4Df>();
		result->m_size = 1;
		result->m_values[0] = val;
		return result;
	}

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

	//////////////////////////////////////////////////////////////////////////
	//	Uniform Base
	//////////////////////////////////////////////////////////////////////////
	void Uniform<bool>::SetUniform()
	{
		switch(m_size)
		{
		case 1:
			glUniform1i(m_uniformLoc, m_values[0]);
			break;
		case 2:
			glUniform2i(m_uniformLoc, m_values[0], m_values[1]);
			break;
		case 3:
			glUniform3i(m_uniformLoc, m_values[0], m_values[1], m_values[2]);
			break;
		case 4:
			glUniform4i(m_uniformLoc, m_values[0], m_values[1], m_values[2], m_values[3]);
			break;
		}
	}

	void Uniform<int>::SetUniform()
	{
		switch(m_size)
		{
		case 1:
			glUniform1i(m_uniformLoc, m_values[0]);
			break;
		case 2:
			glUniform2i(m_uniformLoc, m_values[0], m_values[1]);
			break;
		case 3:
			glUniform3i(m_uniformLoc, m_values[0], m_values[1], m_values[2]);
			break;
		case 4:
			glUniform4i(m_uniformLoc, m_values[0], m_values[1], m_values[2], m_values[3]);
			break;
		}
	}

	void Uniform<unsigned int>::SetUniform()
	{
		switch(m_size)
		{
		case 1:
			glUniform1ui(m_uniformLoc, m_values[0]);
			break;
		case 2:
			glUniform2ui(m_uniformLoc, m_values[0], m_values[1]);
			break;
		case 3:
			glUniform3ui(m_uniformLoc, m_values[0], m_values[1], m_values[2]);
			break;
		case 4:
			glUniform4ui(m_uniformLoc, m_values[0], m_values[1], m_values[2], m_values[3]);
			break;
		}
	}

	void Uniform<float>::SetUniform()
	{
		switch(m_size)
		{
		case 1:
			glUniform1f(m_uniformLoc, m_values[0]);
			break;
		case 2:
			glUniform2f(m_uniformLoc, m_values[0], m_values[1]);
			break;
		case 3:
			glUniform3f(m_uniformLoc, m_values[0], m_values[1], m_values[2]);
			break;
		case 4:
			glUniform4f(m_uniformLoc, m_values[0], m_values[1], m_values[2], m_values[3]);
			break;
		}
	}

	void Uniform<double>::SetUniform()
	{
		switch(m_size)
		{
		case 1:
			glUniform1d(m_uniformLoc, m_values[0]);
			break;
		case 2:
			glUniform2d(m_uniformLoc, m_values[0], m_values[1]);
			break;
		case 3:
			glUniform3d(m_uniformLoc, m_values[0], m_values[1], m_values[2]);
			break;
		case 4:
			glUniform4d(m_uniformLoc, m_values[0], m_values[1], m_values[2], m_values[3]);
			break;
		}
	}

	void Uniform<Matrix4Df>::SetUniform()
	{
		glUniformMatrix4fv(m_uniformLoc, 1, GL_FALSE, m_values[0].Data());
	}

	void Uniform<GLTexture*>::SetUniform()
	{
		if(m_values[0])
		{
			glEnable(GL_TEXTURE_2D);
			glUniform1i(m_uniformLoc, m_size);
			glActiveTexture(GL_TEXTURE0 + m_size);
			glBindTexture(GL_TEXTURE_2D, m_values[0]->GetTextureID());
			glActiveTexture(GL_TEXTURE0);
			glDisable(GL_TEXTURE_2D);
		}
	}
}
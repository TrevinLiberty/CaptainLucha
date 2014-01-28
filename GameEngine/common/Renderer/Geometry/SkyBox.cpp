/**************************************************************************/
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
 *	@file	SkyBox.cpp
 *	@brief	
 *
/****************************************************************************/

#include "SkyBox.h"

#include "Renderer/RendererUtils.h"
#include "Renderer/Renderer.h"
#include "Renderer/Shader/GLProgram.h"
#include "Renderer/Texture/GLTexture.h"

namespace CaptainLucha
{
    SkyBox::SkyBox(
        const char* x1Path, 
        const char* x2Path, 
        const char* y1Path, 
        const char* y2Path, 
        const char* z1Path, 
        const char* z2Path)
        : m_sphere(500.0f, 2)
    {
        glGenTextures(1, &m_textureID);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_textureID);

        const char* texturePaths[] =
        {
            x1Path, x2Path,
            y1Path, y2Path,
            z1Path, z2Path
        };

        GLenum textureTypes[] =
        {
            GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
            GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
            GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
        };

        unsigned int width, height;
        unsigned int* texels;

        for(int i = 0; i < 6; ++i)
        {
            LoadTextureFromFile(texturePaths[i], width, height, texels);

            glTexImage2D(textureTypes[i], 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, texels);

            CleanPreviousTextureLoad();
        }

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

        m_program = new GLProgram("Data/Shaders/Skybox.vert", "Data/Shaders/Skybox.frag");
   
        m_cubeMap = new GLTexture(m_textureID, width, height);

        m_program->SetUnifromTexture("cubeMap", m_cubeMap, true);
    }

    SkyBox::~SkyBox()
    {

    }

    void SkyBox::Draw(Renderer& renderer)
    {
        glEnable(GL_DEPTH_TEST);
        glDepthMask(false);
        g_MVPMatrix->PushMatrix();
        {
            g_MVPMatrix->LoadIdentity();
            g_MVPMatrix->Translate(m_camPos);
            g_MVPMatrix->Scale(20.0f, 20.0f, 20.0f);
            
            m_program->SetUniform(
                "isReflectPass", 
                renderer.GetCurrentPass() == CL_REFLECT_PASS);

            GLint OldCullFaceMode;
            glGetIntegerv(GL_CULL_FACE_MODE, &OldCullFaceMode);
            GLint OldDepthFuncMode;
            glGetIntegerv(GL_DEPTH_FUNC, &OldDepthFuncMode);

            glCullFace(GL_FRONT);
            glDepthFunc(GL_LEQUAL);

            m_sphere.Draw(*m_program);

            glCullFace(OldCullFaceMode); 
            glDepthFunc(OldDepthFuncMode);
        }
        g_MVPMatrix->PopMatrix();
        glDepthMask(true);
    }
}
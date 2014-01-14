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

#include "RendererUtils.h"

#include "Math/Vector4D.h"
#include "Math/Vector3D.h"
#include "Math/Vector2D.h"
#include "Math/Matrix3D.h"

#include "Shader/GLProgram.h"
#include "DeferredRenderer.h"
#include "Font/Font.h"
#include "Geometry/Sphere.h"
#include "Utils/UtilDebug.h"

#include <cmath>
#include <vector>
#include <FreeImage.h>
#include <glfw3.h>
#include <gl/GLU.h>

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Globals
	//////////////////////////////////////////////////////////////////////////
	MatrixStack* g_MVPMatrix;
	DeferredRenderer* g_DeferredRenderer;
	Font* g_DebugFont;

	std::vector<float>* g_DrawVertices;
	std::vector<float>* g_DrawTextureCoords;
	std::vector<float>* g_DrawColor;
	std::vector<float>* g_DrawNormals;

	Sphere* g_DebugDrawSphere;

	bool g_HasDrawBegun = false;
	bool g_IsInitialized = false;
	DrawType g_Drawtype;

	GLFWwindow* g_currentWindow;

	GLProgram* g_CurrentProgram;
	GLProgram* g_DefaultProgram;

	Color g_CurrentColor;
	float g_PointSize;

	int g_VertexLocation;
	int g_NormalLocation;
	int g_TexCoordLocation;
	int g_ColorLocation;

	void IsRendererInitialized()
	{
		REQUIRES(g_IsInitialized && "Render isn't initialized!")
	}

	//////////////////////////////////////////////////////////////////////////
	//	Functions
	//////////////////////////////////////////////////////////////////////////
	bool InitRenderer(bool fullscreen)
	{
		if(!glfwInit())
		{
			return false;
		}
		
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

		GLFWmonitor* primary = NULL;
		if(fullscreen)
			primary = glfwGetPrimaryMonitor();

		//glfwWindowHint(GLFW_SAMPLES, 4);

		g_currentWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "", primary, NULL);
		if(!g_currentWindow)
			return false;

		SetWindowPos(100, 100);

		glfwMakeContextCurrent(g_currentWindow);

		glfwSetInputMode(g_currentWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

		FreeImage_Initialise();
		glewInit();

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glEnable(GL_BLEND);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_POINT_SMOOTH);

		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDepthFunc(GL_LEQUAL);

		g_MVPMatrix = new MatrixStack();
		g_MVPMatrix->Othographic(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 1, -1);
		g_MVPMatrix->Perspective(45.0f, WINDOW_ASPECT_RATIO, 0.1f, 10000.0f);

		g_DebugFont = new Font("Data/Font/Anonymous Pro.ttf", DEBUG_FONT_HEIGHT);

		g_CurrentColor = Vector4Df(1.0f, 1.0f, 1.0f, 1.0f);

		glfwSwapInterval(0);

		g_IsInitialized = true;

		g_DrawVertices = new std::vector<float>();
		g_DrawTextureCoords = new std::vector<float>();
		g_DrawColor = new std::vector<float>();
		g_DrawNormals = new std::vector<float>();

		g_DefaultProgram = new GLProgram("Data/Shaders/SimpleShader.vert", "Data/Shaders/SimpleShader.frag");
		g_CurrentProgram = g_DefaultProgram;

		g_VertexLocation   = -1;
		g_NormalLocation   = -1;
		g_ColorLocation    = -1;
		g_TexCoordLocation = -1;

		g_DebugDrawSphere = new Sphere(1.0f, 4);

		return true;
	}

	void CloseGLFWWindow()
	{
		glfwDestroyWindow(g_currentWindow);
	}

	void SetWindowSize(int width, int height)
	{
		glfwSetWindowSize(g_currentWindow, width, height);
	}

	void SetWindowPos(int x, int y)
	{
		glfwSetWindowPos(g_currentWindow, x, y);
	}

	void DeleteRenderer()
	{
		delete g_MVPMatrix;

		g_IsInitialized = false;

		delete g_DrawVertices;
		delete g_DrawTextureCoords;
		delete g_DrawColor;

		glfwTerminate();
	}

	void HUDMode(bool enable)
	{
		if(enable)
		{
			glDisable(GL_CULL_FACE);
			glEnable(GL_BLEND);

			g_MVPMatrix->SetProjectionMode(CL_ORTHOGRAPHIC);
		}
		else
		{
			glEnable(GL_DEPTH_TEST);
			glEnable(GL_CULL_FACE);

			glEnable(GL_BLEND);

			g_MVPMatrix->SetProjectionMode(CL_PROJECTION);
		}
	}

	void SetCursorHidden(bool val)
	{
		if(val)
			glfwSetInputMode(g_currentWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		else
			glfwSetInputMode(g_currentWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	void EnableVertexAttrib(int loc, int size, int glType, bool normalize, int stride, const void* offset)
	{
		if(loc >= 0)
		{
			glEnableVertexAttribArray(loc);
			glVertexAttribPointer(loc, size, glType, normalize, stride, offset);
		}
	}

	void DisableVertexAttrib(int loc)
	{
		if(loc >= 0)
		{
			glDisableVertexAttribArray(loc);
		}
	}

	void Draw2DDebugText(const Vector2Df& pos, const char* text)
	{
		g_DebugFont->Draw2D(pos, text);
	}

	void GLError(const char* file, int line)
	{
		GLenum errorCode = glGetError();

		std::stringstream errorSS;
		while(errorCode != GL_NO_ERROR)
		{
			const GLubyte* errorString = gluErrorString(errorCode);

			errorSS << file << "(" << line << ")" << " GL ERROR " << errorCode << ": " << "(" << errorString << ")\n";

			errorCode = glGetError();
		}

		if(!errorSS.str().empty())
		{
			trace("\n***********OpengGL Error*************")
			trace(errorSS.str())
			DebugBreak();
		}
	}

	void GetAttributeLocations()
	{
		g_VertexLocation = g_CurrentProgram->GetAttributeLocation("vertex");
		g_NormalLocation = g_CurrentProgram->GetAttributeLocation("normal");
		g_ColorLocation = g_CurrentProgram->GetAttributeLocation("vertexColor");
		g_TexCoordLocation = g_CurrentProgram->GetAttributeLocation("texCoord");
	}

	void ResetAttributeLocations()
	{
		g_VertexLocation   = -1;
		g_NormalLocation   = -1;
		g_ColorLocation    = -1;
		g_TexCoordLocation = -1;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Drawing Methods
	//////////////////////////////////////////////////////////////////////////
	void AddToVector(std::vector<float>& vect, const Vector2Df& vert)
	{
		vect.push_back(vert.x);
		vect.push_back(vert.y);
	}

	void AddToVector(std::vector<float>& vect, const Vector3Df& vert)
	{
		vect.push_back(vert.x);
		vect.push_back(vert.y);
		vect.push_back(vert.z);
	}

	void AddToVector(std::vector<float>& vect, const Vector4Df& vert)
	{
		vect.push_back(vert.x);
		vect.push_back(vert.y);
		vect.push_back(vert.z);
		vect.push_back(vert.w);
	}

	void DrawTriangles(std::vector<float>& pos, std::vector<float>& texCoords, std::vector<float>& color, std::vector<float>& normals)
	{
		GetAttributeLocations();
		if(!texCoords.empty() && texCoords.size() / 2 == pos.size() / 3)
		{
			EnableVertexAttrib(g_TexCoordLocation, 2, GL_FLOAT, GL_FALSE, 0, texCoords.data());
			g_CurrentProgram->SetUniform("useTexture", true);
		}
		else
		{
			g_CurrentProgram->SetUniform("useTexture", false);
		}

		if(!normals.empty() && normals.size() / 3 == pos.size() / 3)
		{
			EnableVertexAttrib(g_NormalLocation, 3, GL_FLOAT, GL_FALSE, 0, normals.data());
		}

		if(!color.empty() && color.size() / 4 == pos.size() / 3)
		{
			EnableVertexAttrib(g_ColorLocation, 4, GL_FLOAT, GL_FALSE, 0, color.data());
			g_CurrentProgram->SetUniform("useVertColor", true);
		}
		else
			g_CurrentProgram->SetUniform("useVertColor", false);

		if(!pos.empty())
		{
			EnableVertexAttrib(g_VertexLocation, 3, GL_FLOAT, GL_FALSE, 0, pos.data());
		}

		g_CurrentProgram->UseProgram();

		glDrawArrays(GL_TRIANGLES, 0, pos.size() / 3);

		DisableVertexAttrib(g_VertexLocation);
		DisableVertexAttrib(g_NormalLocation);
		DisableVertexAttrib(g_TexCoordLocation);
		DisableVertexAttrib(g_ColorLocation);

		ResetAttributeLocations();
	}

	void DrawQuads()
	{
		if(g_DrawVertices->size() < 4 * 3)
			return;

		//Create verts to draw triangle
		std::vector<float> vertices;
		for(size_t i = 0; i < g_DrawVertices->size(); i += 12)
		{
			Vector3Df one((*g_DrawVertices)[i], (*g_DrawVertices)[i + 1], (*g_DrawVertices)[i + 2]);
			Vector3Df two((*g_DrawVertices)[i + 3], (*g_DrawVertices)[i + 4], (*g_DrawVertices)[i + 5]);
			Vector3Df three((*g_DrawVertices)[i + 6], (*g_DrawVertices)[i + 7], (*g_DrawVertices)[i + 8]);
			Vector3Df four((*g_DrawVertices)[i + 9], (*g_DrawVertices)[i + 10], (*g_DrawVertices)[i + 11]);

			AddToVector(vertices, one);
			AddToVector(vertices, two);
			AddToVector(vertices, three);
			AddToVector(vertices, one);
			AddToVector(vertices, three);
			AddToVector(vertices, four);
		}

		//Create verts to draw triangle
		std::vector<float> normals;
		for(size_t i = 0; i < g_DrawNormals->size(); i += 12)
		{
			Vector3Df one((*g_DrawNormals)[i], (*g_DrawNormals)[i + 1], (*g_DrawNormals)[i + 2]);
			Vector3Df two((*g_DrawNormals)[i + 3], (*g_DrawNormals)[i + 4], (*g_DrawNormals)[i + 5]);
			Vector3Df three((*g_DrawNormals)[i + 6], (*g_DrawNormals)[i + 7], (*g_DrawNormals)[i + 8]);
			Vector3Df four((*g_DrawNormals)[i + 9], (*g_DrawNormals)[i + 10], (*g_DrawNormals)[i + 11]);

			AddToVector(normals, one);
			AddToVector(normals, two);
			AddToVector(normals, three);
			AddToVector(normals, one);
			AddToVector(normals, three);
			AddToVector(normals, four);
		}

		std::vector<float> texCoord;
		//Create tex coords to draw triangle
		if(g_DrawTextureCoords->size() >= 4 * 2) //4 verts * 2 uv
		{
			for(size_t i = 0; i < g_DrawTextureCoords->size(); i += 8)
			{
				Vector2Df one((*g_DrawTextureCoords)[i], (*g_DrawTextureCoords)[i + 1]);
				Vector2Df two((*g_DrawTextureCoords)[i + 2], (*g_DrawTextureCoords)[i + 3]);
				Vector2Df three((*g_DrawTextureCoords)[i + 4], (*g_DrawTextureCoords)[i + 5]);
				Vector2Df four((*g_DrawTextureCoords)[i + 6], (*g_DrawTextureCoords)[i + 7]);

				AddToVector(texCoord, one);
				AddToVector(texCoord, two);
				AddToVector(texCoord, three);
				AddToVector(texCoord, one);
				AddToVector(texCoord, three);
				AddToVector(texCoord, four);
			}
		}

		std::vector<float> color;
		//Create tex coords to draw triangle
		if(g_DrawColor->size() >= 4 * 4) //4 verts * 2 uv
		{
			for(size_t i = 0; i < g_DrawColor->size(); i += 16)
			{
				Vector4Df one((*g_DrawColor)[i], (*g_DrawColor)[i + 1], (*g_DrawColor)[i + 2], (*g_DrawColor)[i + 3]);
				Vector4Df two((*g_DrawColor)[i + 4], (*g_DrawColor)[i + 5], (*g_DrawColor)[i + 6], (*g_DrawColor)[i + 7]);
				Vector4Df three((*g_DrawColor)[i + 8], (*g_DrawColor)[i + 9], (*g_DrawColor)[i + 10], (*g_DrawColor)[i + 11]);
				Vector4Df four((*g_DrawColor)[i + 12], (*g_DrawColor)[i + 13], (*g_DrawColor)[i + 14], (*g_DrawColor)[i + 15]);

				AddToVector(color, one);
				AddToVector(color, two);
				AddToVector(color, three);
				AddToVector(color, one);
				AddToVector(color, three);
				AddToVector(color, four);
			}
		}

		DrawTriangles(vertices, texCoord, color, normals);
	}

	void DrawPoints()
	{
		GetAttributeLocations();
		g_CurrentProgram->SetUniform("color", g_CurrentColor);

		if(!g_DrawVertices->empty())
		{
			g_CurrentProgram->SetUniform("useTexture", false);
			g_CurrentProgram->SetUniform("pointSize", g_PointSize);
			EnableVertexAttrib(g_VertexLocation, 3, GL_FLOAT, GL_FALSE, 0, g_DrawVertices->data());
		}
		else
			return;

		if(!g_DrawColor->empty() && g_DrawColor->size() / 4 == g_DrawVertices->size() / 3)
		{
			EnableVertexAttrib(g_ColorLocation, 4, GL_FLOAT, GL_FALSE, 0, g_DrawColor->data());
			g_CurrentProgram->SetUniform("useVertColor", true);
		}
		else
			g_CurrentProgram->SetUniform("useVertColor", false);

		g_CurrentProgram->UseProgram();
		glDrawArrays(GL_POINTS, 0, g_DrawVertices->size() / 3);

		DisableVertexAttrib(g_VertexLocation);
		DisableVertexAttrib(g_ColorLocation);

		ResetAttributeLocations();
	}

	void DrawLines()
	{
		GetAttributeLocations();
		g_CurrentProgram->SetUniform("color", g_CurrentColor);

		if(!g_DrawVertices->empty())
		{
			EnableVertexAttrib(g_VertexLocation, 3, GL_FLOAT, GL_FALSE, 0, g_DrawVertices->data());
			g_CurrentProgram->SetUniform("useTexture", false);
		}
		else
			return;

		if(!g_DrawColor->empty())
		{
			EnableVertexAttrib(g_ColorLocation, 4, GL_FLOAT, GL_FALSE, 0, g_DrawColor->data());
			g_CurrentProgram->SetUniform("useVertColor", true);
		}
		else
			g_CurrentProgram->SetUniform("useVertColor", false);

		g_CurrentProgram->UseProgram();
		glDrawArrays(GL_LINES, 0, g_DrawVertices->size() / 3);

		DisableVertexAttrib(g_VertexLocation);
		DisableVertexAttrib(g_ColorLocation);

		ResetAttributeLocations();
	}

	void DrawBegin(DrawType type)
	{
		IsRendererInitialized();

		g_HasDrawBegun = true;

		g_Drawtype = type;

		g_DrawVertices->clear();
		g_DrawTextureCoords->clear();
		g_DrawColor->clear();
		g_DrawNormals->clear();
	}

	void DrawEnd()
	{
		IsRendererInitialized();

 		g_CurrentProgram->SetModelViewProjection();
		g_CurrentProgram->SetUniform("color", g_CurrentColor);

		if(!g_DrawVertices->empty() && g_HasDrawBegun)
		{
			switch(g_Drawtype)
			{
			case CL_QUADS:
				DrawQuads();
				break;
			case CL_TRIANGLES:
				DrawTriangles(*g_DrawVertices, *g_DrawTextureCoords, *g_DrawColor, *g_DrawNormals);
				break;
			case CL_POINTS:
				DrawPoints();
				break;
			case CL_LINES:
				DrawLines();
				break;
			}
		}

		g_DrawVertices->clear();
		g_DrawTextureCoords->clear();
		g_DrawColor->clear();
		g_DrawNormals->clear();

		g_HasDrawBegun = false;

		g_CurrentProgram->ClearTextureUnits();
	}

	void SetColor(float r, float g, float b, float a)
	{
		g_CurrentColor = Vector4Df(r, g, b, a);
	}

	void SetColor(const Vector4Df& color)
	{
		g_CurrentColor = color;
	}

	void SetColor(const Color& color)
	{
		g_CurrentColor = color;
	}

	void SetColor(const Color& color, float alphaOverride)
	{
		g_CurrentColor = Color(color.r, color.g, color.b, alphaOverride);
	}

	void clSetPointSize(float size)
	{
		g_PointSize = size;
	}

	void clVertex3f(float x, float y, float z)
	{
		if(g_HasDrawBegun)
		{
			g_DrawVertices->push_back(x);
			g_DrawVertices->push_back(y);
			g_DrawVertices->push_back(z);
		}
	}

	void clVertex3f(const Vector3Df& vert)
	{
		if(g_HasDrawBegun)
		{
			g_DrawVertices->push_back(vert.x);
			g_DrawVertices->push_back(vert.y);
			g_DrawVertices->push_back(vert.z);
		}
	}

	void clNormal3(float nx, float ny, float nz)
	{
		if(g_HasDrawBegun)
		{
			g_DrawNormals->push_back(nx);
			g_DrawNormals->push_back(ny);
			g_DrawNormals->push_back(nz);
		}
	}

	void clTexCoord(float u, float v)
	{
		if(g_HasDrawBegun)
		{
			g_DrawTextureCoords->push_back(u);
			g_DrawTextureCoords->push_back(v);
		}
	}

	void clColor4(float r, float g, float b, float a)
	{
		if(g_HasDrawBegun)
		{
			g_DrawColor->push_back(r);
			g_DrawColor->push_back(g);
			g_DrawColor->push_back(b);
			g_DrawColor->push_back(a);
		}
	}

	void clColor4(const Color& color)
	{
		clColor4(color.r, color.g, color.b, color.a);
	}

	void FullScreenPass()
	{
		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->LoadIdentity();
		g_MVPMatrix->SetProjectionMode(CL_ORTHOGRAPHIC);
		DrawBegin(CL_QUADS);
		{
			clVertex3(0.0f, 0.0f, 0.0f);
			clVertex3(WINDOW_WIDTHf, 0.0f, 0.0f);
			clVertex3(WINDOW_WIDTHf, WINDOW_HEIGHTf, 0.0f);
			clVertex3(0.0f, WINDOW_HEIGHTf, 0.0f);

			clTexCoord(0, 0);
			clTexCoord(1, 0);
			clTexCoord(1, 1);
			clTexCoord(0, 1);
		}
		DrawEnd();
		g_MVPMatrix->SetProjectionMode(CL_PROJECTION);
		g_MVPMatrix->PopMatrix();
	}

	void DrawPlus(const Vector3Df& pos, float size)
	{
		DrawPlus(pos.x, pos.y, pos.z, size);
	}

	void DrawPlus(float x, float y, float z, float size)
	{
		DrawBegin(CL_LINES);
		clVertex3(x - size, y, z);
		clVertex3(x + size, y, z);

		clVertex3(x, y - size, z);
		clVertex3(x, y + size, z);

		clVertex3(x, y, z - size);
		clVertex3(x, y, z + size);
		DrawEnd();
	}

	void DrawArrow(const Vector3Df& from, const Vector3Df& to)
	{
		SetColor(1.0f, 1.0f, 1.0f, g_CurrentColor.a);

		DrawBegin(CL_LINES);
		clColor4(1.0f, 0.0f, 0.0f, g_CurrentColor.a);
		clVertex3(from.x, from.y, from.z);
		clColor4(0.0f, 1.0f, 0.0f, g_CurrentColor.a);
		clVertex3(to.x, to.y, to.z);
		DrawEnd();

		SetColor(0.0f, 1.0f, 0.0f, g_CurrentColor.a);
		DrawPlus(to, 0.1f);
	}

	void DrawAxis(float size)
	{
		glLineWidth(5.0f);
		g_MVPMatrix->PushMatrix();

		SetColor(1.0f, 0.0f, 0.0f, 1.0f);
		DrawBegin(CL_LINES);
		clVertex3(0.0f, 0.0f, 0.0f);
		clVertex3(size, 0.0f, 0.0f);
		DrawEnd();

		SetColor(0.0f, 1.0f, 0.0f, 1.0f);
		DrawBegin(CL_LINES);
		clVertex3(0.0f, 0.0f, 0.0f);
		clVertex3(0.0f, size, 0.0f);
		DrawEnd();

		SetColor(0.0f, 0.0f, 1.0f, 1.0f);
		DrawBegin(CL_LINES);
		clVertex3(0.0f, 0.0f, 0.0f);
		clVertex3(0.0f, 0.0f, size);
		DrawEnd();

		g_MVPMatrix->PopMatrix();
		glLineWidth(1.0f);
	}

	void DrawFilledCircle(const Vector2Df& pos, float spaceing, float radius)
	{
		float radSpacing = DegreesToRadians(spaceing);

		std::vector<Vector2Df> verts;
		verts.push_back(pos);

		for(float i = 0; i <= 2 * 3.1415f; i += radSpacing)
		{
			verts.push_back(Vector2Df(pos.x + cos(i) * radius,				pos.y + sin(i) * radius));
			verts.push_back(Vector2Df(pos.x + cos(i + radSpacing) * radius,	pos.y + sin(i + radSpacing) * radius));
		}

		//g_currentShader->SetModelViewProjection();
		//g_currentShader->SetUniform("color", 4, &g_CurrentColor[0]);
		//g_currentShader->UseProgram();

		static int vl = 0;////g_currentShader->GetAttributeLocation("vertex");

		glEnableVertexAttribArray(vl);
		glVertexAttribPointer(vl, 2, GL_FLOAT, GL_FALSE, 0, verts.data());
		glDrawArrays(GL_TRIANGLE_FAN, 0, verts.size());
		glDisableVertexAttribArray(vl);
	}

	void DrawLineCircle(const Vector2Df& pos, float spaceing, float radius)
	{
		float radSpacing = DegreesToRadians(spaceing);

		DrawBegin(CL_LINES);
		for(float i = 0; i <= 2 * 3.1415f; i += radSpacing)
		{
			clVertex3(pos.x + cos(i) * radius,				pos.y + sin(i) * radius, 0.0f);
			clVertex3(pos.x + cos(i + radSpacing) * radius,	pos.y + sin(i + radSpacing) * radius, 0.0f);
		}
		DrawEnd();
	}

	void DrawLine(const Vector2Df& l1, const Vector2Df& l2)
	{
		DrawBegin(CL_LINES);
		clVertex3(l1.x, l1.y, 0.0f);
		clVertex3(l2.x, l2.y, 0.0f);
		DrawEnd();
	}

	void DrawLine(const Vector3Df& l1, const Vector3Df& l2)
	{
		DrawBegin(CL_LINES);
		clVertex3(l1.x, l1.y, l1.z);
		clVertex3(l2.x, l2.y, l2.z);
		DrawEnd();
	}

	void SetWindowTitle(const std::string& title)
	{
		glfwSetWindowTitle(g_currentWindow, title.c_str());
	}

	void SetTexture(const std::string& name, GLTexture* glTexture)
	{
		g_CurrentProgram->SetUnifromTexture(name, glTexture);
	}

	void SetUniform(const char* name, bool val)
	{
		g_CurrentProgram->SetUniform(name, val);
	}

	void SetUniform(const char* name, int val)
	{
		g_CurrentProgram->SetUniform(name, val);
	}

	void SetUniform(const char* name, float val)
	{
		g_CurrentProgram->SetUniform(name, val);
	}

	void SetUniform(const char* name, Color val)
	{
		g_CurrentProgram->SetUniform(name, val);
	}

	void SetGLProgram(GLProgram* program)
	{
		if(program == NULL)
			g_CurrentProgram = g_DefaultProgram;
		else
			g_CurrentProgram = program;

		GetAttributeLocations();
	}

	GLProgram* GetCurrentProgram()
	{
		return g_CurrentProgram;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Debug Drawing
	//////////////////////////////////////////////////////////////////////////
	void DrawDebugPlus(const Vector3Df& pos, float size)
	{
		float currentAlpha = g_CurrentColor.a;

		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		g_CurrentColor.a = .1f;
		DrawPlus(pos.x, pos.y, pos.z, size);

		glEnable(GL_DEPTH_TEST);
		g_CurrentColor.a = currentAlpha;
		DrawPlus(pos.x, pos.y, pos.z, size);
	}

	void DrawDebugArrow(const Vector3Df& from, const Vector3Df& to)
	{
		float currentAlpha = g_CurrentColor.a;

		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		g_CurrentColor.a = 0.5f;
		DrawArrow(from, to);

		glEnable(GL_DEPTH_TEST);
		g_CurrentColor.a = currentAlpha;
		DrawArrow(from, to);
	}

	void DrawDebugBBox(const Vector3Df& pos, const Vector3Df& dimensions)
	{
		float vertices[] =
		{
			-1.0f,  1.0f,  1.0f,
			-1.0f, -1.0f,  1.0f,
			 1.0f, -1.0f,  1.0f,
			 1.0f,  1.0f,  1.0f,

			 1.0f,  1.0f, -1.0f,
			 1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f,  1.0f, -1.0f,

			 1.0f, -1.0f,  1.0f,
			 1.0f, -1.0f, -1.0f,
			 1.0f,  1.0f, -1.0f,
			 1.0f,  1.0f,  1.0f,

			-1.0f,  1.0f,  1.0f,
			-1.0f,  1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f,  1.0f,

			 1.0f,  1.0f,  1.0f, 
			 1.0f,  1.0f, -1.0f, 
			-1.0f, 1.0f, -1.0f, 
			-1.0f, 1.0f,  1.0f, 

			 1.0f, -1.0f, -1.0f,
			 1.0f, -1.0f,  1.0f,
			-1.0f, -1.0f,  1.0f,
			-1.0f, -1.0f, -1.0f,
		};

		const float hx = dimensions.x * 0.5f;
		const float hy = dimensions.y * 0.5f;
		const float hz = dimensions.z * 0.5f;

		for(int i = 0; i < 72; i += 3)
		{
			vertices[i] *= hx;
			vertices[i + 1] *= hy;
			vertices[i + 2] *= hz;

			vertices[i] += pos.x;
			vertices[i + 1] += pos.y;
			vertices[i + 2] += pos.z;
		}

		float currentAlpha = g_CurrentColor.a;

		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		g_CurrentColor.a = .1f;
		DrawBegin(CL_LINES);
			for(int i = 0; i < 6; ++i)
			{
				const int ROW_INDEX = i * 12;
				for(int j = 0; j < 4; ++j)
				{
					const int index1 = ROW_INDEX + j * 3;
					const int index2 = ROW_INDEX + (j == 3 ? 0 : (j + 1) * 3);

					clVertex3(vertices[index1], vertices[index1 + 1], vertices[index1 + 2]);
					clVertex3(vertices[index2], vertices[index2 + 1], vertices[index2 + 2]);
				}
			}
		DrawEnd();

		glEnable(GL_DEPTH_TEST);
		g_CurrentColor.a = currentAlpha;
		DrawBegin(CL_LINES);
		for(int i = 0; i < 6; ++i)
		{
			const int ROW_INDEX = i * 12;
			for(int j = 0; j < 4; ++j)
			{
				const int index1 = ROW_INDEX + j * 3;
				const int index2 = ROW_INDEX + (j == 3 ? 0 : (j + 1) * 3);

				clVertex3(vertices[index1], vertices[index1 + 1], vertices[index1 + 2]);
				clVertex3(vertices[index2], vertices[index2 + 1], vertices[index2 + 2]);
			}
		}
		DrawEnd();

		g_CurrentColor.a = .05f;
		DrawBegin(CL_QUADS);
		for(int i = 0; i < 72; i += 3)
		{
			clVertex3(vertices[i], vertices[i+1], vertices[i+2]);
		}
		DrawEnd();
	}

	void DrawDebugSphere(const Vector3Df& pos, float radius)
	{
		g_DefaultProgram->SetUniform("color", g_CurrentColor);

		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->Translate(pos);
		g_MVPMatrix->Scale(radius, radius, radius);
		g_DebugDrawSphere->Draw(*g_DefaultProgram);
		g_MVPMatrix->PopMatrix();
	}

	//////////////////////////////////////////////////////////////////////////
	//	Advanced Shader Methods
	//////////////////////////////////////////////////////////////////////////
	void SetTangentSpaceMatrix(TangentSpaceVertex& p0, const TangentSpaceVertex& p1, const TangentSpaceVertex& p2, bool setNormals_) 
	{
		Vector3Df v1(p1.x_ - p0.x_, p1.y_ - p0.y_, p1.z_ - p0.z_);
		Vector3Df v2(p2.x_ - p0.x_, p2.y_ - p0.y_, p2.z_ - p0.z_);

		Vector2Df t1(p1.u_ - p0.u_, p1.v_ - p0.v_);
		Vector2Df t2(p2.u_ - p0.u_, p2.v_ - p0.v_);

		const float DET = 1.0f / (t1.x * t2.y - t2.x * t1.y);
		
		Vector3Df tangent = (v1 * t2.y - v2 * t1.y) * DET;
		Vector3Df biTangent = (v1 * -t2.x + v2 * t1.x) * DET;

		tangent.Normalize();
		biTangent.Normalize();

		p0.tstx_ = tangent.x;
		p0.tsty_ = tangent.y;
		p0.tstz_ = tangent.z;

		p0.tsbx_ = biTangent.x;
		p0.tsby_ = biTangent.y;
		p0.tsbz_ = biTangent.z;

		if(setNormals_)
		{
			p0.nx_ = biTangent.y * tangent.z - biTangent.z * tangent.y;
			p0.ny_ = biTangent.z * tangent.x - biTangent.x * tangent.z;
			p0.nz_ = biTangent.x * tangent.y - biTangent.y * tangent.x;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Utils
	//////////////////////////////////////////////////////////////////////////
	Justification GetJustEnumFromString(const std::string& just)
	{
		if (_strcmpi(just.c_str(), "center") == 0)
		{
			return CL_CENTER;
		}
		else if (_strcmpi(just.c_str(), "left") == 0)
		{
			return CL_LEFT;
		}
		else if (_strcmpi(just.c_str(), "right") == 0)
		{
			return CL_RIGHT;
		}
		else if (_strcmpi(just.c_str(), "top") == 0)
		{
			return CL_TOP;
		}
		else if (_strcmpi(just.c_str(), "bottom") == 0)
		{
			return CL_BOTTOM;
		}

		return CL_CENTER;
	}

	void AddQuadCubeToArray(std::vector<Vector3Df>& verts, const Vector3Df& min, const Vector3Df& max)
	{
		//x
		verts.push_back(Vector3Df(min.x, max.y, max.z));
		verts.push_back(Vector3Df(min.x, max.y, min.z));
		verts.push_back(Vector3Df(min.x, min.y, min.z));
		verts.push_back(Vector3Df(min.x, min.y, max.z));

		verts.push_back(Vector3Df(max.x, min.y, max.z));
		verts.push_back(Vector3Df(max.x, min.y, min.z));
		verts.push_back(Vector3Df(max.x, max.y, min.z));
		verts.push_back(Vector3Df(max.x, max.y, max.z));

 		//y
 		verts.push_back(Vector3Df(max.x, min.y, min.z));
 		verts.push_back(Vector3Df(max.x, min.y, max.z));
 		verts.push_back(Vector3Df(min.x, min.y, max.z));
 		verts.push_back(Vector3Df(min.x, min.y, min.z));
 
		verts.push_back(Vector3Df(max.x, max.y, max.z));
		verts.push_back(Vector3Df(max.x, max.y, min.z));
		verts.push_back(Vector3Df(min.x, max.y, min.z));
		verts.push_back(Vector3Df(min.x, max.y, max.z));
 
 		//z
		verts.push_back(Vector3Df(max.x, max.y, min.z));
		verts.push_back(Vector3Df(max.x, min.y, min.z));
		verts.push_back(Vector3Df(min.x, min.y, min.z));
		verts.push_back(Vector3Df(min.x, max.y, min.z));
 
		verts.push_back(Vector3Df(min.x, max.y, max.z));
		verts.push_back(Vector3Df(min.x, min.y, max.z));
		verts.push_back(Vector3Df(max.x, min.y, max.z));
		verts.push_back(Vector3Df(max.x, max.y, max.z));
	}

	Vertex& Vertex::operator= (const TangentSpaceVertex& rhs)
	{
		x_ = rhs.x_;
		y_ = rhs.y_;
		z_ = rhs.z_;
		nx_ = rhs.nx_;
		ny_ = rhs.ny_;
		nz_ = rhs.nz_;
		u_ = rhs.u_,
		v_ = rhs.v_;
		return *this;
	}
}
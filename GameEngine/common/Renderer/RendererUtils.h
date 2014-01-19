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
 *	@file	RendererUtils.h
 *	@brief	
 *
/****************************************************************************/

#ifndef RENDERER_UTILS_H_CL
#define RENDERER_UTILS_H_CL

#define GL_ERROR() GLError(__FILE__, __LINE__);

#include "MatrixStack.h"
#include "Color.h"
#include "Texture/GLTexture.h"
#include "Shader/GLProgram.h"

#include "Math/Vector2D.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	//Global ModelViewProjection Matrix
	//
	extern MatrixStack* g_MVPMatrix;

	static const int WINDOW_WIDTH = 1620;
	static const int WINDOW_HEIGHT = 920;

	static const float WINDOW_WIDTHf = float(WINDOW_WIDTH);
	static const float WINDOW_HEIGHTf = float(WINDOW_HEIGHT);

	static const float WINDOW_HALF_WIDTH = WINDOW_WIDTH / 2.0f;
	static const float WINDOW_HALF_HEIGHT = WINDOW_HEIGHT / 2.0f;

	static const int DEBUG_FONT_HEIGHT = 14;
	static const float WINDOW_ASPECT_RATIO = static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT;

	struct Vertex;
	struct TangentSpaceVertex;
	enum Justification;
	enum DrawType;

	bool InitRenderer(bool fullscreen);
	void DeleteRenderer();

	void CloseGLFWWindow();
	void SetWindowSize(int width, int height);
	void SetWindowPos(int x, int y);

	void HUDMode(bool enable);

	void SetCursorHidden(bool val);

	void EnableVertexAttrib(int loc, int size, int glType, bool normalize, int stride, const void* offset);
	void DisableVertexAttrib(int loc);

	void Draw2DDebugText(const Vector2Df& pos, const char* text);

	void GLError(const char* file, int line);

	//////////////////////////////////////////////////////////////////////////
	//	Drawing Methods
	//////////////////////////////////////////////////////////////////////////
	void DrawBegin(DrawType type);
	void DrawEnd();

	void SetUtilsColor(float r, float g, float b, float a);
	void SetUtilsColor(const Vector4Df& color);
	void SetUtilsColor(const Color& color);
	void SetUtilsColor(const Color& color, float alphaOverride);
	void clSetPointSize(float size);

	void clVertex3f(float x, float y, float z);
	void clVertex3f(const Vector3Df& vert);

	template<typename Tx, typename Ty, typename Tz>
	void clVertex3(Tx x, Ty y, Tz z)
	{
		clVertex3f((float)x, (float)y, (float)z);
	}

	void clNormal3(float nx, float ny, float nz);
	void clTexCoord(float u, float v);
	void clColor4(float r, float g, float b, float a);
	void clColor4(const Color& color);

	void FullScreenPass();

	void DrawPlus(const Vector3Df& pos, float size);
	void DrawPlus(float x, float y, float z, float size);

	void DrawArrow(const Vector3Df& from, const Vector3Df& to);

	void DrawAxis(float size);

	void DrawFilledCircle(const Vector2Df& pos, float spaceing, float radius);
	void DrawLineCircle(const Vector2Df& pos, float spaceing, float radius);
	void DrawLine(const Vector2Df& l1, const Vector2Df& l2);
	void DrawLine(const Vector3Df& l1, const Vector3Df& l2);

	void SetWindowTitle(const std::string& title);

	void SetTexture(const std::string& name, GLTexture* glTexture);

	void SetUniform(const char* name, bool val);
	void SetUniform(const char* name, int val);
	void SetUniform(const char* name, float val);
	void SetUniform(const char* name, const Color& val);
    void SetUniform(const char* name, const Vector3Df& val);

	//Setting to NULL reverts to the default shader
	void SetGLProgram(GLProgram* program);
	GLProgram* GetCurrentProgram();

	//////////////////////////////////////////////////////////////////////////
	//	Debug Drawing
	//////////////////////////////////////////////////////////////////////////
	void DrawDebugPlus(const Vector3Df& pos, float size);
	void DrawDebugArrow(const Vector3Df& from, const Vector3Df& to);

	void DrawDebugBBox(const Vector3Df& pos, const Vector3Df& dimensions);
	void DrawDebugSphere(const Vector3Df& pos, float radius);

	//////////////////////////////////////////////////////////////////////////
	//	Advanced Shader Methods
	//////////////////////////////////////////////////////////////////////////
	void SetTangentSpaceMatrix(TangentSpaceVertex& p0, const TangentSpaceVertex& p1, const TangentSpaceVertex& p2, bool setNormals_ = true);

	//////////////////////////////////////////////////////////////////////////
	// Utils
	//////////////////////////////////////////////////////////////////////////
	Justification GetJustEnumFromString(const std::string& just);

	void AddQuadCubeToArray(std::vector<Vector3Df>& verts, const Vector3Df& min, const Vector3Df& max);

	/**
	 * @brief     Pushes x and y to back of vect. Extra types added to avoid having to cast stuff.
	 * @param	  std::vector<Tv> & vect
	 * @param	  const Tx & x
	 * @param	  const Ty & y
	 * @return    void
	 */
	template <typename Tv, typename Tx, typename Ty>
	inline void AddToVector(std::vector<Tv>& vect, const Tx& x, const Ty& y)
	{
		vect.push_back((Tv)x);
		vect.push_back((Tv)y);
	}

	/**
	 * @brief     Pushes x, y and z to back of vect. Extra types added to avoid having to cast stuff.
	 * @param	  std::vector<Tv> & vect
	 * @param	  const Tx & x
	 * @param	  const Ty & y
	 * @param	  const Tz & z
	 * @return    void
	 */
	template <typename Tv, typename Tx, typename Ty, typename Tz>
	inline void AddToVector(std::vector<Tv>& vect, const Tx& x, const Ty& y, const Tz& z)
	{
		vect.push_back((Tv)x);
		vect.push_back((Tv)y);
		vect.push_back((Tv)z);
	}

	/**
	 * @brief     Pushes x, y, z and w to back of vect. Extra types added to avoid having to cast stuff.
	 * @param	  std::vector<Tv> & vect
	 * @param	  const Tx & x
	 * @param	  const Ty & y
	 * @param	  const Tz & z
	 * @param	  const Tw & w
	 * @return    void
	 */
	template <typename Tv, typename Tx, typename Ty, typename Tz, typename Tw>
	inline void AddToVector(std::vector<Tv>& vect, const Tx& x, const Ty& y, const Tz& z, const Tw& w)
	{
		vect.push_back((Tv)x);
		vect.push_back((Tv)y);
		vect.push_back((Tv)z);
		vect.push_back((Tv)w);
	}

	enum DrawType
	{
		CL_QUADS,
		CL_TRIANGLES,
		CL_POINTS,
		CL_LINES
	};

	enum Justification
	{
		CL_CENTER,
		CL_LEFT,
		CL_RIGHT,
		CL_TOP,
		CL_BOTTOM
	};

	struct Vert2D
	{
		float x_, y_;		//Pos
		float u_, v_;			//TexCoord

		Vert2D() {}
		Vert2D(float x, float y, float u, float v)
			: x_(x), y_(y), u_(u), v_(v) {}
		Vert2D(float u, float v)
			: u_(u), v_(v) {}
	};

	struct Vert2DColor
	{
		float x_, y_;		//Pos
		float u_, v_;			//TexCoord
		float r_, g_, b_, a_;

		Vert2DColor() {}
		Vert2DColor(float x, float y, float u, float v)
			: x_(x), y_(y), u_(u), v_(v) {}
		Vert2DColor(float u, float v)
			: u_(u), v_(v) {}

		void SetColor(const Color& color)
		{
			r_ = color.r;
			g_ = color.g;
			b_ = color.b;
			a_ = color.a;
		}
	};

	struct TangentSpaceVertex;

	struct Vertex
	{
		float x_, y_, z_;	
		float nx_, ny_, nz_;
		float u_, v_;		

		Vertex& operator= (const TangentSpaceVertex& rhs);
	};

	struct TangentSpaceVertex
	{
		float  x_,  y_,  z_;
		float nx_, ny_, nz_;
		float u_, v_;
		float tstx_, tsty_, tstz_;
		float tsbx_, tsby_, tsbz_;

		bool operator== (const TangentSpaceVertex& rhs) const
		{
			return (abs(x_ - rhs.x_) <  0.00000001 
				&& abs(y_ - rhs.y_) < 0.00000001
				&& abs(z_ - rhs.z_) < 0.00000001);
		}


		TangentSpaceVertex& operator= (const Vertex& rhs)
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

		TangentSpaceVertex& operator+ (const Vector3Df& translate)
		{
			x_ += translate.x;
			y_ += translate.y;
			z_ += translate.z;
			return *this;
		}

		TangentSpaceVertex& operator* (const float& scale)
		{
			x_ *= scale;
			y_ *= scale;
			z_ *= scale;
			return *this;
		}
	};
}

#endif
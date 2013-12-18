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
 *	@file	ClusteredRenderer.h
 *	@brief	
 *
/****************************************************************************/

#ifndef CLUSTEREDRENDERER_H_CL
#define CLUSTEREDRENDERER_H_CL

#include "DeferredRenderer.h"

namespace CaptainLucha
{
	class Cuda_Clustered;

	class ClusteredRenderer : public DeferredRenderer
	{
	public:
		ClusteredRenderer();
		~ClusteredRenderer();

		void Draw();

		//TODO. Need to update when FOV changes.
		void UpdateScreenDimensions();

		//Debug
		//
		int GetNumberOfUsedClusters() const;

		void DebugDraw();

		//Clustered Constants
		//
		static const int TILE_SIZE_X = 64;
		static const int TILE_SIZE_Y = 64;

		static const int MAX_NUM_LIGHTS = 1024;

	protected:
		void InitRenderer();

		void SetAssigmentUniforms();
		void AssignClusters();
		void ResetFlagBuffer();

	private:
		float m_invNear;
		float m_invFarFunc;
		float m_debugTime;

		int m_numTilesX;
		int m_numTilesY;
		int m_numTilesZ;

		bool m_isClusterInit;

		Cuda_Clustered*	m_cudaClustered;
		GLProgram* m_clusterAssignment;

		GLTexture* m_clusterFlagTexture;
		GLTexture* m_clusterGridTexture;
		GLTexture* m_clusterLightTexture;

		unsigned int m_clusterFlagBuffer;
		unsigned int m_clusterGridBuffer;
		unsigned int m_clusterLightBuffer;

		unsigned int m_clusterBufferSize;

		std::vector<Vector3Df> m_debugPoints;
	};
}

#endif
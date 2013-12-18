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

#include "ClusteredRenderer.h"
#include "CUDA/Cuda_Clustered.h"
#include "RendererUtils.h"
#include "Input/InputSystem.h"

namespace CaptainLucha
{
	ClusteredRenderer::ClusteredRenderer()
		: DeferredRenderer(),
		  m_isClusterInit(false),
		  m_debugTime(0.0f)
	{
		InitRenderer();
	}

	ClusteredRenderer::~ClusteredRenderer()
	{

	}

	void ClusteredRenderer::Draw()
	{
		m_debugTime -= 1 / 60.0f;

		PopulateGBuffers();
		AssignClusters();

		if(m_debugDraw)
			DebugRenderGBuffer();
	}

	void ClusteredRenderer::UpdateScreenDimensions()
	{
		m_numTilesX = (WINDOW_WIDTH + TILE_SIZE_X - 1) / TILE_SIZE_X;
		m_numTilesY = (WINDOW_HEIGHT + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
		m_numTilesZ = 256;

		m_invNear = 1.0f / 0.1f;

		m_invFarFunc = 2.0f * tanf(DegreesToRadians(45.0f) * 0.5f) / float(m_numTilesY);
		m_invFarFunc = 1.0f / logf(m_invFarFunc + 1);
	}

	int ClusteredRenderer::GetNumberOfUsedClusters() const
	{
		return m_cudaClustered->GetNumUsedClusters();
	}

	void ClusteredRenderer::DebugDraw()
	{
		g_MVPMatrix->SetProjectionMode(CL_PROJECTION);
		SetGLProgram(NULL);
		SetColor(Color::White);
		SetUniform("useAlpha", false);
		SetUniform("useTexture", false);
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);
		DrawBegin(CL_POINTS);
		for(int i = 0; i < m_debugPoints.size(); ++i)
			clVertex3f(m_debugPoints[i].x, m_debugPoints[i].y, m_debugPoints[i].z);
		DrawEnd();
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
	}

	void ClusteredRenderer::SetAssigmentUniforms()
	{
		REQUIRES(m_clusterAssignment != NULL)

		m_clusterAssignment->SetUniform("NUM_TILES_X", m_numTilesX);//26
		m_clusterAssignment->SetUniform("NUM_TILES_Y", m_numTilesY);//15
		m_clusterAssignment->SetUniform("NUM_TILES_Z", m_numTilesZ);//256
		m_clusterAssignment->SetUniform("TileSizeX", TILE_SIZE_X);//64
		m_clusterAssignment->SetUniform("TileSizeY", TILE_SIZE_Y);//64
		m_clusterAssignment->SetUniform("invNear", m_invNear);//10
		m_clusterAssignment->SetUniform("invFarFunc", m_invFarFunc);//18.6
	}

	void ClusteredRenderer::InitRenderer()
	{
		REQUIRES(!m_isClusterInit)

		UpdateScreenDimensions();

		//Create Textures
		//
		unsigned int flagTex;
		glGenTextures(1, &flagTex);

		GL_ERROR()

		//Create Buffers
		//
		m_clusterBufferSize = m_numTilesX * m_numTilesY * m_numTilesZ;

		glGenBuffers(1, &m_clusterFlagBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, m_clusterFlagBuffer);
		glBufferData(GL_ARRAY_BUFFER, m_clusterBufferSize * sizeof(unsigned int), NULL, GL_DYNAMIC_COPY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		//Attach Textures to Buffers
		//
		glBindTexture(GL_TEXTURE_BUFFER, flagTex);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, m_clusterFlagBuffer);
		m_clusterFlagTexture  = new GLTexture(flagTex, -1, -1);
		glBindTexture(GL_TEXTURE_BUFFER, 0);

		//Set buffer to zero
		glBindBuffer(GL_ARRAY_BUFFER, m_clusterFlagBuffer);
		unsigned int* p = reinterpret_cast<unsigned int*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
		memset(p, 0, m_clusterBufferSize * sizeof(unsigned int));
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		GL_ERROR()

		m_clusterAssignment = new GLProgram("Data/Shaders/ClusterAssignment.vert", "Data/Shaders/ClusterAssignment.frag");
		SetAssigmentUniforms();

		m_isClusterInit = true;

		m_cudaClustered = Cuda_Clustered::CreateRenderer();
		m_cudaClustered->Init(m_clusterFlagBuffer, m_numTilesX, m_numTilesY, m_numTilesZ);

		//std::vector<float> temp;
		//temp.reserve(10000);
		//for(float i = 0.0; i <= 1.0; i += 0.001)
		//{
		//	float d = i * 10000.0f + 0.1f;
		//	int t = log(d * m_invNear) * m_invFarFunc;
		//	temp.push_back(t);
		//}
	}

	void ClusteredRenderer::AssignClusters()
	{
		//Disable Color Writing
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

		//Bind Image Buffer
		glBindImageTexture(0, m_clusterFlagTexture->GetTextureID(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
		GL_ERROR();

		//Full screen pass
		SetGLProgram(m_clusterAssignment);
		SetTexture("Depth", m_depth);
		FullScreenPass();
		SetGLProgram(NULL);

 		//Wait for shader to write assignment data
		glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

		//Unbind ImageBuffer
		glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
		GL_ERROR();

		//Enable Color Writing
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

		m_cudaClustered->FindUniqueClusters();

		if(InputSystem::GetInstance()->IsKeyDown(GLFW_KEY_MINUS) && m_debugTime <= 0.0f)
		{
			const unsigned int* UNIQUE_CLUSTER_INDICES = m_cudaClustered->GetUniqueIndices();
			const unsigned int NUM_UNIQUE_CLUSTERS = m_cudaClustered->GetNumUsedClusters();

			m_debugPoints.clear();
			m_debugPoints.reserve(NUM_UNIQUE_CLUSTERS);

			std::stringstream ss;
			for(int i = 0; i < NUM_UNIQUE_CLUSTERS; ++i)
			{
				const unsigned int INDEX = UNIQUE_CLUSTER_INDICES[i];
				Vector3Df ClusterIndex;
				ClusterIndex.x = INDEX % m_numTilesX;
				ClusterIndex.y = (INDEX / m_numTilesX) % m_numTilesY;
				ClusterIndex.z = INDEX / (m_numTilesX * m_numTilesY);

				m_debugPoints.push_back(Vector3Df());
				Vector3Df& newPoint = m_debugPoints.back();

				newPoint.x = ClusterIndex.x;
				newPoint.y = ClusterIndex.y;
				newPoint.z = ClusterIndex.z;//powf(10, ClusterIndex.z / m_invFarFunc) / m_invNear;

				ss << newPoint.z << " ";
			}
			trace(ss.str())

			m_debugTime = 2.5f;
		}
	}
}


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

#include "CUDA_Clustered.h"
#include "CUDA_Utils.h"
#include "CudaHelpers\helper_cuda.h"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_new.h>

#include "CUDA_Clustered_Kernels.cuh"
#include "CUDA_Lights.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "chag/pp/prefix.cuh"

#include <windows.h>
#include <cuda_gl_interop.h>

namespace CaptainLucha
{
	class Cuda_Clustered_Implementation : public Cuda_Clustered
	{
	public:
		Cuda_Clustered_Implementation()
			: m_hIsInitialized(false),
			  m_hUniqueIndices(NULL),
			  m_hnumUsedCluters(0),
		      m_uniqueClusterSize(0),
			  m_lightKeyBufferSize(0) {};
		~Cuda_Clustered_Implementation();

		void Init(
			unsigned int clusterAssignmentBuffer, 
			unsigned int numClustersX, 
			unsigned int numClustersY, 
			unsigned int numClustersZ);

		void FindUniqueClusters();
		void CreateLightBVH(int numLights, float* lightPosRadData);

		unsigned int GetNumUsedClusters() const {return m_hnumUsedCluters;}
		const unsigned int* GetUniqueIndices() const {return m_hUniqueIndices;}

	protected:
		void SetUniqueBuffersSize(unsigned int size);
		void SetLightsBufferSize(unsigned int size);

	private:
		//Host Variables
		//
		bool m_hIsInitialized;

		int m_hmaxNumClusters;

		unsigned int m_hnumUsedCluters;
		unsigned int* m_hUniqueIndices;

		unsigned int m_numClustersX;
		unsigned int m_numClustersY;
		unsigned int m_numClustersZ;

		//Device Variables
		//
		cudaGraphicsResource* m_dflagResource;

		//Find Unique Clusters
		unsigned int  m_uniqueClusterSize;
		thrust::device_ptr<unsigned int> m_dUniqueIndices;
		thrust::device_ptr<unsigned int> m_dUniqueKeys;
		
		thrust::device_ptr<unsigned int> m_dprefixResource;
		thrust::device_ptr<unsigned int> m_dnumUsedClusters;

		//Assign Lights to Clusters
		thrust::device_ptr<float4>		 m_dminAABB;
		thrust::device_ptr<float4>		 m_dmaxAABB;

		unsigned int  m_lightKeyBufferSize;

		thrust::device_ptr<float4>		 m_dlightPosRadData;
		thrust::device_ptr<unsigned int> m_dmortonLightKeys;
		thrust::device_ptr<unsigned int> m_dlightIndices;

		//Light BVH
		LightBVH* m_lightBVH;
	};

	Cuda_Clustered_Implementation::~Cuda_Clustered_Implementation()
	{

	}

	template<typename T>
	void GetDeviceMappedPointer(cudaGraphicsResource* resource, T** outMappedPointer)
	{
		cudaGraphicsMapResources(1, &resource);

		void* result = NULL;
		unsigned int bufferSize = 0;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&result, &bufferSize, resource));
		*outMappedPointer = reinterpret_cast<T*>(result);
	}

	void UnmapDevicePointer(cudaGraphicsResource* resource)
	{
		cudaGraphicsUnmapResources(1, &resource);
	}

	Cuda_Clustered* Cuda_Clustered::CreateRenderer()
	{
		if(!IsCudaInit())
			InitCuda();

		return new Cuda_Clustered_Implementation();
	}

	void Cuda_Clustered_Implementation::Init(
		unsigned int clusterAssignmentBuffer, 
		unsigned int numClustersX, 
		unsigned int numClustersY, 
		unsigned int numClustersZ)
	{
		m_numClustersX = numClustersX;
		m_numClustersY = numClustersY;
		m_numClustersZ = numClustersZ;
		
		m_hmaxNumClusters = m_numClustersX * m_numClustersY * m_numClustersZ;

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_dflagResource, clusterAssignmentBuffer, cudaGraphicsRegisterFlagsNone));

		m_dprefixResource  = thrust::device_malloc<unsigned int>(m_hmaxNumClusters * sizeof(unsigned int));
		m_dnumUsedClusters = thrust::device_malloc<unsigned int>(sizeof(unsigned int));

		m_dminAABB = thrust::device_malloc<float4>(sizeof(float4));
		m_dmaxAABB = thrust::device_malloc<float4>(sizeof(float4));

		SetUniqueBuffersSize(1024);
		SetLightsBufferSize(128);

		m_lightBVH = new LightBVH();
		m_lightBVH->SetNumberOfLights(128);

		m_hIsInitialized = true;
	}

	void Cuda_Clustered_Implementation::SetUniqueBuffersSize(unsigned int size)
	{
		if(size > m_uniqueClusterSize)
		{
			m_uniqueClusterSize = size + 128;

			thrust::device_free(m_dUniqueIndices);
			thrust::device_free(m_dUniqueKeys);

			m_dUniqueIndices = thrust::device_malloc<unsigned int>(m_uniqueClusterSize * sizeof(unsigned int));
			m_dUniqueKeys    = thrust::device_malloc<unsigned int>(m_uniqueClusterSize * sizeof(unsigned int));
		}
	}

	void Cuda_Clustered_Implementation::SetLightsBufferSize(unsigned int size)
	{
		if(size > m_lightKeyBufferSize)
		{
			m_lightKeyBufferSize = size + 128;
			
			thrust::device_free(m_dlightPosRadData);
			thrust::device_free(m_dmortonLightKeys);
			thrust::device_free(m_dlightIndices);

			m_dlightPosRadData = thrust::device_malloc<float4>(m_lightKeyBufferSize * sizeof(float4));
			m_dmortonLightKeys = thrust::device_malloc<unsigned int>(m_lightKeyBufferSize * sizeof(unsigned int));
			m_dlightIndices    = thrust::device_malloc<unsigned int>(m_lightKeyBufferSize * sizeof(unsigned int));;
		}
	}

	////////////////////////////////////////////////////////////
	//
	//	1. Get flag resource pointer from device.
	//	2. Compute prefix sum on the flag array.
	//	3. Using the Flag array and the output from the prefix sum we 
	//		can create an array of indices with an unique set of flagged clusters.
	//
	////////////////////////////////////////////////////////////
	void Cuda_Clustered_Implementation::FindUniqueClusters()
	{
		unsigned int* flagPointer = NULL;
		GetDeviceMappedPointer(m_dflagResource, &flagPointer);

		chag::pp::prefix(flagPointer, flagPointer + m_hmaxNumClusters, m_dprefixResource.get(), m_dnumUsedClusters.get());

		cudaMemcpy(&m_hnumUsedCluters, m_dnumUsedClusters.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost);

		if(m_hnumUsedCluters)
		{
			SetUniqueBuffersSize(m_hnumUsedCluters);

			cudaMemset(m_dUniqueIndices.get(), 0, m_hnumUsedCluters * sizeof(unsigned int));

			const int NUM_BLOCKS = GetReqNumBlocks(m_hmaxNumClusters);
			GetUniqueClusters<<<NUM_BLOCKS, DEFAULT_THREADS_PER_BLOCK>>>(
				m_hmaxNumClusters, 
				m_numClustersX, m_numClustersY, m_numClustersZ,
				flagPointer, m_dprefixResource.get(), m_dUniqueIndices.get());

			delete m_hUniqueIndices;
			m_hUniqueIndices = new unsigned int[m_hnumUsedCluters];
			cudaMemcpy(m_hUniqueIndices, m_dUniqueIndices.get(), m_hnumUsedCluters * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		}

		//Reset flag buffer
		cudaMemset(flagPointer, 0, m_hmaxNumClusters * sizeof(unsigned int));

		UnmapDevicePointer(m_dflagResource);
	}

	void Cuda_Clustered_Implementation::CreateLightBVH(int numLights, float* lightPosRadData)
	{
		//Increase buffer sizes if needed
		SetLightsBufferSize(numLights);

		//copy light data to device
		cudaMemcpy(m_dlightPosRadData.get(), lightPosRadData, numLights * sizeof(float4), cudaMemcpyHostToDevice);

		if(numLights)
		{
			//Update BVH tp support numLights
			m_lightBVH->SetNumberOfLights(numLights);

			/**********************************/
			//Create BVH Leaf Nodes
			/**********************************/
			//Compute the AABB the encompasses all lights
			ComputeLightAABB(
				m_dlightPosRadData.get(), m_dlightPosRadData.get() + numLights, 
				m_dminAABB.get(), m_dmaxAABB.get());
		
			//Using the AABB, calculate the Morton Code for all lights.
			//	- allows us to quickly sort the lights based on position. Getting ready to create the BVH
            //morton code - https://www.marcusbannerman.co.uk/index.php/home/39-websitepages/108-morton-ordering.html
			ComputeLightMortonCodes(
				m_dlightPosRadData.get(), m_dminAABB.get(), m_dmaxAABB.get(), 
				numLights, m_dmortonLightKeys.get(), m_dlightIndices.get());

			//Sort the Lights based on their Morton Code.
			//	- Put in "Morton Order"
			thrust::stable_sort_by_key(m_dmortonLightKeys, m_dmortonLightKeys + numLights, m_dlightIndices);

			/**********************************/
			//Create Rest of BVH
			/**********************************/
            cudaMemset(m_lightBVH->m_minAABBNodes.get(), 0, 1024 * sizeof(unsigned int));

            //Start with leaf nodes and create bottom parents of tree first
			const int CURRENT_BRANCHING_FACTOR = LightBVH::BRANCHING_FACTOR;//todo update this when factor can change
			dim3 threadsPerBlock = dim3(CURRENT_BRANCHING_FACTOR, 6, 1);
			dim3 blocks = dim3((int)std::ceil(numLights / 6.0f), 1, 1);
			SetLeafsAndCalculateParentNodes<<<blocks, threadsPerBlock>>>(m_dlightPosRadData.get(), *m_lightBVH, m_dlightIndices.get());

            //fill in the rest of the tree starting from the bottom and moving up.
            //  levels index starts at 0 -> root node
            //  level for leaf -> numLevels - 1
            //we filled in the parent nodes above leafs (numLevels - 2)
            //now fill in the parent nodes above that (numLevels - 3) until root
            for(int i = m_lightBVH->GetNumLevels() - 3; i >= 0; --i)
            {
                const unsigned int NUM_NODES_ON_LEVEL = powf(CURRENT_BRANCHING_FACTOR, i);
                dim3 threadsPerBlock = dim3(CURRENT_BRANCHING_FACTOR, 6, 1);
			    dim3 blocks = dim3((int)std::ceil(NUM_NODES_ON_LEVEL / 6.0f), 1, 1);
                CalculateParentNodes<<<blocks, threadsPerBlock>>>(*m_lightBVH, i);
            }

			float4* temp = new float4[m_lightBVH->GetNumNonLeafNodes()];
			cudaMemcpy(temp, m_lightBVH->m_maxAABBNodes.get(), m_lightBVH->GetNumNonLeafNodes() * sizeof(float4), cudaMemcpyDeviceToHost);
 			delete temp;
		}
	}
}
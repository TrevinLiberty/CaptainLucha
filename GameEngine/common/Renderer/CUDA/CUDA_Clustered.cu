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

#include "Utils/UtilDebug.h"
#include "CUDA_Clustered.h"
#include "CUDA_Utils.h"
#include "CudaHelpers\helper_cuda.h"

#include "CUDA_Clustered_Kernels.cu"

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
			  m_hnumUsedCluters(0),
		      m_uniqueClusterSize(360),
			  m_hUniqueIndices(NULL) {};
		~Cuda_Clustered_Implementation() {};

		void Init(
			unsigned int clusterFlagBuffer, 
			unsigned int numClustersX, 
			unsigned int numClustersY, 
			unsigned int numClustersZ);

		void FindUniqueClusters();

		unsigned int GetNumUsedClusters() const {return m_hnumUsedCluters;}

		const unsigned int* GetUniqueIndices() const {return m_hUniqueIndices;}

	protected:
		void SetUniqueBuffersSize(unsigned int size);

	private:
		//Host Variables
		bool m_hIsInitialized;

		int m_hmaxNumClusters;
		unsigned int m_hnumUsedCluters;

		unsigned int m_numClustersX;
		unsigned int m_numClustersY;
		unsigned int m_numClustersZ;

		unsigned int* m_hUniqueIndices;

		//Device Variables
		cudaGraphicsResource* m_dflagResource;
		cudaGraphicsResource* m_dclusterGridResource;

		unsigned int m_uniqueClusterSize;
		unsigned int* m_dUniqueIndices;
		unsigned int* m_dUniqueKeys;
		
		unsigned int* m_dprefixResource;
		unsigned int* m_dnumUsedClusters;
	};

	template<typename T>
	void GetDeviceMappedPointer(cudaGraphicsResource* resource, T** outMappedPointer)
	{
		void* result = NULL;
		unsigned int bufferSize = 0;
		cudaGraphicsResourceGetMappedPointer(&result, &bufferSize, resource);
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
		
		checkCudaErrors(cudaMalloc((void**)&m_dprefixResource, m_hmaxNumClusters * sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**)&m_dnumUsedClusters, 1 * sizeof(unsigned int)));

		checkCudaErrors(cudaMalloc((void**)&m_dUniqueIndices, (m_uniqueClusterSize * sizeof(unsigned int))));
		checkCudaErrors(cudaMalloc((void**)&m_dUniqueKeys,    (m_uniqueClusterSize * sizeof(unsigned int))));

		m_hIsInitialized = true;
	}

	void Cuda_Clustered_Implementation::SetUniqueBuffersSize(unsigned int size)
	{
		if(size > m_uniqueClusterSize)
		{
			checkCudaErrors(cudaFree((void*)m_dUniqueIndices));
			checkCudaErrors(cudaFree((void*)m_dUniqueKeys));

			checkCudaErrors(cudaMalloc((void**)&m_dUniqueIndices, (size * sizeof(unsigned int))));
			checkCudaErrors(cudaMalloc((void**)&m_dUniqueKeys,    (size * sizeof(unsigned int))));

			m_uniqueClusterSize = size;
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
		cudaGraphicsMapResources(1, &m_dflagResource);

		unsigned int* flagPointer = NULL;
		GetDeviceMappedPointer(m_dflagResource, &flagPointer);

		chag::pp::prefix(flagPointer, flagPointer + m_hmaxNumClusters, m_dprefixResource, m_dnumUsedClusters);

		cudaMemcpy(&m_hnumUsedCluters, m_dnumUsedClusters, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		if(m_hnumUsedCluters)
		{
			SetUniqueBuffersSize(m_hnumUsedCluters);

			cudaMemset(m_dUniqueIndices, 0, m_hnumUsedCluters * sizeof(unsigned int));

			const int NUM_THREADS_PER_BLOCK = 192;
			const int NUM_BLOCKS = std::ceil(m_hmaxNumClusters / float(NUM_THREADS_PER_BLOCK));
			GetUniqueClusters<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(
				m_hmaxNumClusters, 
				m_numClustersX, m_numClustersY, m_numClustersZ,
				flagPointer, m_dprefixResource, m_dUniqueIndices);

			delete m_hUniqueIndices;
			m_hUniqueIndices = new unsigned int[m_hmaxNumClusters];
			cudaMemcpy(m_hUniqueIndices, m_dUniqueIndices, m_hnumUsedCluters * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		}

		//Reset flag buffer
		cudaMemset(flagPointer, 0, m_hmaxNumClusters * sizeof(unsigned int));

		UnmapDevicePointer(m_dflagResource);
	}
}
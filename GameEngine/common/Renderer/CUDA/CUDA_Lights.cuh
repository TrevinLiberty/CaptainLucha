#ifndef CUDA_LIGHTS_H
#define CUDA_LIGHTS_H

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "vector_types.h"
#include "CudaHelpers\helper_math.h"

namespace CaptainLucha
{
	class LightBVH;

	void ComputeLightAABB(
		const float4* lightPosRadStart, 
		const float4* lightPosRadEnd, 
		float4* outAABBMin, float4* outAABBMax);

	void ComputeLightMortonCodes(
		const float4* lightPosRad, 
		const float4* AABBMin, 
		const float4* AABBMax,
		int numLights,
		unsigned int* outKeys,
		unsigned int* outIndices);

	__global__ void __launch_bounds__(32*6,6) 
	SetLeafsAndCalculateParentNodes(
		const float4* lightPosRad, 
		const LightBVH bvh, 
		const unsigned int* sortedLightIndices);

    __global__ void __launch_bounds__(32*6,6) 
    CalculateParentNodes(
		const LightBVH bvh,
        int currentLevel);

	//////////////////////////////////////////////////////////////////////////
	//	Light BVH
    //
    //  +Branching Factor set to 32. (b/c 32 threads in a warp)
	//		change to float3 for AABB?
	//		Update to be able to change branching factor on the fly
	//////////////////////////////////////////////////////////////////////////
	class LightBVH
	{
	public:
		LightBVH() 
			: m_numLights(0),
			  m_lightBufferSize(0),
			  m_numLevels(0),
			  m_numNonLeafNodes(0) {}
		~LightBVH() {};

		static const int BRANCHING_FACTOR = 32;
		static const float INV_LOG_BRANCH;

		inline __host__ void SetNumberOfLights(unsigned int numLights);
		inline __device__ __host__ unsigned int GetNumLights() const {return m_numLights;}

		inline __device__ __host__ unsigned int GetNumNonLeafNodes() const {return m_numNonLeafNodes;}
		inline __device__ __host__ unsigned int GetNumLevels() const {return m_numLevels;}

		inline __device__ void SetRootNode(const float4* minAABB, const float4* maxAABB) const
		{
			*m_minAABBNodes.get() = *minAABB;
			*m_maxAABBNodes.get() = *maxAABB;
		}

		inline __device__ void SetLeaf(unsigned int index, unsigned int lightIndex) const
		{
			if(index < m_numLights)
				m_lightIndicesNodes.get()[index] = lightIndex;
		}

		inline __device__ void SetNode(unsigned int level, unsigned int index, const float3& AABBmin, const float3& AABBmax) const
		{
			const unsigned int LEVEL_INDEX = GetLevelIndex(level);
			const unsigned int NODE_INDEX  = LEVEL_INDEX + index;
			
			m_minAABBNodes.get()[NODE_INDEX] = make_float4(AABBmin, 0.0f);
			m_maxAABBNodes.get()[NODE_INDEX] = make_float4(AABBmax, 0.0f);
		}

        inline __device__ const float4& GetNodeMinAABB(int index) const
        {
            return m_minAABBNodes.get()[index];
        }

        inline __device__ const float4& GetNodeMaxAABB(int index) const
        {
            return m_maxAABBNodes.get()[index];
        }

		//returns the index of the BVH level in a 1D array given the BRANCHING_FACTOR and level needed.
		//Performance. pre cache
		inline __device__ __host__ int GetLevelIndex(int level) const
		{
			return (int)std::floor((pow((float)BRANCHING_FACTOR, (float)level) - 1.0f) / (BRANCHING_FACTOR - 1));
		}

		//Binary tree can have 2^n leaf nodes, where n = height
		//	Same for tree with different branch factors. 32^n ... n = height
		//	logbase32(numLights) == logbase10(numLights) / logbase10(32);
		inline __device__ __host__ int GetNumLayersNeeded(int numLights) const
		{
			return numLights == 0 ? 1 : (int)std::ceil((log((float)numLights) / log((float)BRANCHING_FACTOR)) + 1);
		}

		//Calculates max number of non-leaf nodes to fill numLevels.
		//	notice: can change to min number of non-leaf nodes to support some number of lights
		//		if memory becomes a problem. Other way avoids reallocating memory a lot.
		inline __device__ __host__ int CalcNumNonLeafNodes(int numLevels) const
		{
			return GetLevelIndex(numLevels - 1);
		}

        inline __device__ __host__ int GetNumNodesAtLevel(int level) const
        {
            return powf(BRANCHING_FACTOR, level);
        }

	//private:
		thrust::device_ptr<float4>	     m_minAABBNodes;
		thrust::device_ptr<float4>		 m_maxAABBNodes;
		thrust::device_ptr<unsigned int> m_lightIndicesNodes;

		unsigned int m_numLights;
		unsigned int m_lightBufferSize;
		unsigned int m_numLevels;
		unsigned int m_numNonLeafNodes;
	};

	inline __host__ void LightBVH::SetNumberOfLights(unsigned int numLights)
	{
		const unsigned int NUM_LAYERS_NEEDED = GetNumLayersNeeded(numLights);
		const unsigned int NUM_NON_LEAF_NODES_NEEDED = CalcNumNonLeafNodes(NUM_LAYERS_NEEDED);
		if(m_numNonLeafNodes < NUM_NON_LEAF_NODES_NEEDED)
		{
			thrust::device_free(m_minAABBNodes);
			thrust::device_free(m_maxAABBNodes);

			m_minAABBNodes = thrust::device_malloc<float4>(NUM_NON_LEAF_NODES_NEEDED * sizeof(float4));
			m_maxAABBNodes = thrust::device_malloc<float4>(NUM_NON_LEAF_NODES_NEEDED * sizeof(float4));

			m_numNonLeafNodes = NUM_NON_LEAF_NODES_NEEDED;
			m_numLevels = NUM_LAYERS_NEEDED;
		}

		if(m_lightBufferSize < numLights)
		{
			thrust::device_free(m_lightIndicesNodes);

			m_lightIndicesNodes = thrust::device_malloc<unsigned int>(numLights * sizeof(unsigned int));

			m_lightBufferSize = numLights;
		}

		m_numLights = numLights;
		m_numNonLeafNodes = NUM_NON_LEAF_NODES_NEEDED;
		m_numLevels = NUM_LAYERS_NEEDED;
	}
}

#endif
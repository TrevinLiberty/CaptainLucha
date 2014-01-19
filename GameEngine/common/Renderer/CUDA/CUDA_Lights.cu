#include "CUDA_Lights.cuh"
#include "float.h"
#include "CUDA_Utils.h"
#include "chag\pp\reduce.cuh"

//#include <cmath>

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	ComputeLightAABB
	//////////////////////////////////////////////////////////////////////////
	struct CalcMinAABB
	{
		__device__ __host__ ::float4 operator() ( ::float4 aA, ::float4 aB) const
		{
			return make_float4( 
				min(aA.x-aA.w,aB.x-aB.w), 
				min(aA.y-aA.w,aB.y-aB.w), 
				min(aA.z-aA.w,aB.z-aB.w), 
				0.0f);
		}

		__device__ __host__ ::float4 identity() const
		{
			return make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.0f);
		}
	};

	struct CalcMaxAABB
	{
		__device__ __host__ float4 operator() (float4 aA, float4 aB) const
		{
			return make_float4( 
				max(aA.x+aA.w,aB.x+aB.w), 
				max(aA.y+aA.w,aB.y+aB.w), 
				max(aA.z+aA.w,aB.z+aB.w), 
				0.0f);
		}

		__device__ __host__ float4 identity() const
		{
			return make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f);
		}
	};

	void ComputeLightAABB(
		const float4* lightPosRadStart, 
		const float4* lightPosRadEnd, 
		float4* outAABBMin, float4* outAABBMax)
	{
		chag::pp::reduce(lightPosRadStart, lightPosRadEnd, outAABBMin, CalcMinAABB());
		chag::pp::reduce(lightPosRadStart, lightPosRadEnd, outAABBMax, CalcMaxAABB());
	}

	//////////////////////////////////////////////////////////////////////////
	//	ComputeLightAABB
	//		performance: spreadBits may be slow
	//////////////////////////////////////////////////////////////////////////
	//Found at http://www.cse.chalmers.se/~olaolss/main_frame.php?contents=publication&id=clustered_shading Jan 7th 2014. ClusteredForwardDemo
	template <int BITS>
	__host__ __device__ inline 
		unsigned int spreadBits(unsigned int value, unsigned int stride, unsigned int offset)
	{
		unsigned int x = (unsigned int(1) << BITS) - 1;
		unsigned int v = value & x;
		unsigned int mask = 1;
		unsigned int result = 0;
		for (unsigned int i = 0; i < BITS; ++i)
		{
			result |= mask & v;
			v = v << (stride - 1);
			mask = mask << stride;
		}
		return result << offset;
	}

	//Found at http://www.cse.chalmers.se/~olaolss/main_frame.php?contents=publication&id=clustered_shading Jan 7th 2014. ClusteredForwardDemo
	template <int BITS>
	__host__ __device__ inline 
		unsigned int unspreadBits(unsigned int value, unsigned int stride, unsigned int offset)
	{
		unsigned int v = value >> offset;
		unsigned int mask = 1;
		unsigned int result = 0;
		for (unsigned int i = 0; i < BITS; ++i)
		{
			result |= mask & v;
			v = v >> (stride - 1);
			mask = mask << 1;
		}
		return result;
	}

	//Given a light's position and the center position of the all encompassing AABB.
	//	Calculate the direction between the position. 
	//	Put each axis of the position in the range of 0-256;
	//	Interweave the bits (8) of each axis into a 32 bit unsigned int. (Morton Code)
	__global__ void CalcMortonCode(
		const float4* __restrict__ lightPosRad, 
		const float4* minAABB, const float4* maxAABB,
		int numLights,
		unsigned int* __restrict__ outKeys,
		unsigned int* __restrict__ outIndices)
	{
		const unsigned int INDEX = blockIdx.x * blockDim.x + threadIdx.x;
		if(INDEX < numLights)
		{
			const float3 MINAABB = make_float3(*minAABB);
			const float3 POS	 = make_float3(lightPosRad[INDEX]);
			const float3 RNG	 = make_float3(*maxAABB) - MINAABB;
			const float3 DIF	 = POS - MINAABB;

			const float3 NORM_DIR = DIF / RNG;

			const unsigned int BITS_PER_COORD = 8;
			const unsigned int COORD_RANGE = (1 << BITS_PER_COORD) - 1;
			const unsigned int MORTON_CODE = 
				  spreadBits<BITS_PER_COORD>(unsigned int(NORM_DIR.x * COORD_RANGE), 3, 0)	
				| spreadBits<BITS_PER_COORD>(unsigned int(NORM_DIR.y * COORD_RANGE), 3, 1)
				| spreadBits<BITS_PER_COORD>(unsigned int(NORM_DIR.z * COORD_RANGE), 3, 2);

			outKeys[INDEX]	  = MORTON_CODE;
			outIndices[INDEX] = INDEX;
		}
	}

	void ComputeLightMortonCodes(
		const float4* lightPosRad, 
		const float4* AABBMin, 
		const float4* AABBMax,
		int numLights,
		unsigned int* outKeys,
		unsigned int* outIndices)
	{
		const int NUM_BLOCKS = GetReqNumBlocks(numLights);
		CalcMortonCode<<<NUM_BLOCKS, DEFAULT_THREADS_PER_BLOCK>>>(
			lightPosRad,
			AABBMin, AABBMax,
			numLights,
			outKeys,
			outIndices
			);
	}

	//Operators for SetLeafsAndCalculateNodes(...)
	struct AabbMinOp
	{
		__device__ __host__ float3 operator() (float3 aA, float3 aB) const
		{
			return make_float3( 
				min(aA.x,aB.x), 
				min(aA.y,aB.y), 
				min(aA.z,aB.z) );
		}

		__device__ __host__ float3 identity() const
		{
			return make_float3( +FLT_MAX, +FLT_MAX, +FLT_MAX );
		}
	};

	struct AabbMaxOp
	{
		__device__ __host__ float3 operator() (float3 aA, float3 aB) const
		{
			return make_float3( 
				max(aA.x,aB.x), 
				max(aA.y,aB.y), 
				max(aA.z,aB.z) );
		}

		__device__ __host__ float3 identity() const
		{
			return make_float3( -FLT_MAX, -FLT_MAX, -FLT_MAX );
		}
	};

	//Launch Bounds specifies (maxThreadsPerBlock, minBlocksPerRun). Helps compiler optimize kernel.
	//	threadIdx.x = Current Group's leaf index
	//	threadIdx.y = Thread Leaf Group (Warp) Index
	//	blockIdx.x  = Block Leaf Group Index
	//	
	//	blockDim	= Number of threads in a block
	__global__ void __launch_bounds__(32*6,6) 
	SetLeafsAndCalculateParentNodes(
		const float4* lightPosRad, 
		const LightBVH bvh, 
		const unsigned int* sortedLightIndices)
	{
        //Temp buffer for chag::pp
		struct SharedBuffer
		{
			float3 chagReduceBuffer[32 + 32 / 2];
		};

		__shared__ SharedBuffer sharedBuf[6];
		SharedBuffer& currentBuffer = sharedBuf[threadIdx.y];

		//Index of a group of leafs (lights)
		const unsigned int GROUP_INDEX = threadIdx.y + blockIdx.x * blockDim.y;
		const unsigned int LEAF_INDEX  = GROUP_INDEX * blockDim.x + threadIdx.x;

		//Number of non-leaf nodes minus number of parents to leafs. Plus current group number.
		if(bvh.GetLevelIndex(bvh.GetNumLevels() - 2) + GROUP_INDEX < bvh.GetNumNonLeafNodes())
		{
			unsigned int lightIndex;
			float3 lightAABBMin;
            float3 lightAABBMax;

			if(LEAF_INDEX < bvh.GetNumLights())
			{
				lightIndex = sortedLightIndices[LEAF_INDEX];
				const float4& curPosRad = lightPosRad[lightIndex];

				lightAABBMin = make_float3(
					curPosRad.x - curPosRad.w, 
					curPosRad.y - curPosRad.w, 
					curPosRad.z - curPosRad.w);

				lightAABBMax = make_float3(
					curPosRad.x + curPosRad.w, 
					curPosRad.y + curPosRad.w, 
					curPosRad.z + curPosRad.w);
			}
			else
			{
				lightIndex   = 0;
				lightAABBMin = make_float3(+FLT_MAX, +FLT_MAX, +FLT_MAX);
				lightAABBMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			}

			typedef chag::pp::KernelSetupWarp<> WarpSetup;
			typedef chag::pp::Unit<WarpSetup> WarpUnit;

			float3 nodeMinAABB = WarpUnit::reduce(lightAABBMin, AabbMinOp(), currentBuffer.chagReduceBuffer);
			float3 nodeMaxAABB = WarpUnit::reduce(lightAABBMax, AabbMaxOp(), currentBuffer.chagReduceBuffer);
		
			bvh.SetLeaf(LEAF_INDEX, lightIndex);

			//numlevels - 2 == level above leafs
			bvh.SetNode(bvh.GetNumLevels() - 2, GROUP_INDEX, nodeMinAABB, nodeMaxAABB);
		}
	}

    /*
        Algorithm similar to SetLeafsAndCalculateParentNodes(...) but using nodes
            instead of leafs

        threads per block dim3(32, 6, 1)
        num blocks        dim3(numNodesCurrentLevel / 6, 1, 1)

        MAX_NUM_THREADS = 32 * 6 * (numNodesCurLevel / 6) == 32 * numNodesCurLevel

            - 32 threads per node at current level.
            - each thread represent child node
            - each warp (32 threads) represents ONE parent node
            - each warp calculates the min and max AABB for their parent node.

            + parent node id == threadIdx.y + blockIdx.x * blockDim.y
                -where blockDim.y == threadsPerBlock.y == 6
                -range 0 -> (numNodesCurrentLevel - 1)
                -note, id != BVH array index

            + child node can be found doing:
                -childNodeStartIndex (in BVH array) + parentIndex * blockDim.x + threadIdx.x;
                -where blockDim.x == threadsPerBlock.x == 32
                -where theadIdx.x == 0 -> 31

        *note, threadsPerBlock.y is six to fit more warps in a block. Better optimized.
            -could possibly make 12 (32 * 12 == 384) less than 512 limit on threads per block. TEST
    */
    __global__ void __launch_bounds__(32*6,6) 
    CalculateParentNodes(
		const LightBVH bvh,
        int currentLevel)
	{
        //Temp buffer for chag::pp
		struct SharedBuffer
		{
			float3 chagReduceBuffer[32 + 32 / 2];
		};

		__shared__ SharedBuffer sharedBuf[6];
		SharedBuffer& currentBuffer = sharedBuf[threadIdx.y];

		const unsigned int NODE_INDEX = threadIdx.y + blockIdx.x * blockDim.y;

        if(NODE_INDEX >= bvh.GetNumNodesAtLevel(currentLevel))
            return;

        const int CHILD_INDEX = bvh.GetLevelIndex(currentLevel + 1) + NODE_INDEX * blockDim.x + threadIdx.x;
        
        float3 childMinAABB = make_float3(+FLT_MAX, +FLT_MAX, +FLT_MAX);
        float3 childMaxAABB = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        if(CHILD_INDEX < bvh.GetNumNonLeafNodes())
        {
            childMinAABB = make_float3(bvh.GetNodeMinAABB(CHILD_INDEX));
            childMaxAABB = make_float3(bvh.GetNodeMaxAABB(CHILD_INDEX));
        }

		typedef chag::pp::KernelSetupWarp<> WarpSetup;
		typedef chag::pp::Unit<WarpSetup> WarpUnit;

		float3 nodeMinAABB = WarpUnit::reduce(childMinAABB, AabbMinOp(), currentBuffer.chagReduceBuffer);
		float3 nodeMaxAABB = WarpUnit::reduce(childMaxAABB, AabbMaxOp(), currentBuffer.chagReduceBuffer);

        bvh.SetNode(currentLevel, NODE_INDEX, nodeMinAABB, nodeMaxAABB);
	}

	const float LightBVH::INV_LOG_BRANCH = 1 / log((float)BRANCHING_FACTOR);
}
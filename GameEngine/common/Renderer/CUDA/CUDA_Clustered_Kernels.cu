#ifndef CUDA_CLUSTERED_KERNELS_H
#define CUDA_CLUSTERED_KERNELS_H

namespace CaptainLucha
{
	//	Example
	//	- (clusterFlagBuffer)   [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
	//	- (clusterPrefixResult) [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 6]
	//
	//	INDEX = 2;
	//	if(clusterFlagBuffer[INDEX] == 1)
	//	{
	//		INDEX_INTO_OUTUNIQUECLUSTERINDICES = clusterPrefixResult[INDEX];
	//		outUniqueClusterIndices[INDEX_INTO_OUTUNIQUECLUSTERINDICES] = INDEX;
	//	}
	//
	//	sing the index, we're able to reverse compute the cluster key.
	//
	//
	__global__ void GetUniqueClusters(
		const int maxNumClusters, 
		const int DIMX, const int DIMY, const int DMZ,
		const unsigned int* __restrict__ clusterFlagBuffer, 
		const unsigned int* __restrict__ clusterPrefixResult,
		unsigned int* __restrict__ outUniqueClusterIndices)
	{
		const unsigned int INDEX = blockIdx.x * blockDim.x + threadIdx.x;
		if(INDEX < maxNumClusters && clusterFlagBuffer[INDEX])
		{
			outUniqueClusterIndices[clusterPrefixResult[INDEX]] = INDEX;
		}
	}
}

#endif
#ifndef CUDA_CLUSTERED_KERNELS_H
#define CUDA_CLUSTERED_KERNELS_H

namespace CaptainLucha
{
	__global__ void GetUniqueClusters(
		const int maxNumClusters, 
		const int DIMX, const int DIMY, const int DMZ,
		const unsigned int* __restrict__ clusterFlagBuffer, 
		const unsigned int* __restrict__ clusterPrefixResult,
		unsigned int* __restrict__ outUniqueClusterIndices);
}

#endif
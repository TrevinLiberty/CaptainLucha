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

#include "CUDA_Utils.h"
#include "Utils/Utils.h"
#include "CudaHelpers\helper_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace CaptainLucha
{
	bool g_IsCudaInit = false;
 	cudaDeviceProp g_deviceProperties;
 
 	//Forward declares
 	//
 	void SetBestCudaDevice();
 	int GetMaxGFlopDevice();

	void InitCuda()
	{
 		REQUIRES(!g_IsCudaInit)
 
 		SetBestCudaDevice();
 
 		g_IsCudaInit = true;
	}

	bool IsCudaInit()
	{
		return g_IsCudaInit;
	}

 	void SetBestCudaDevice()
 	{
 		const int deviceID = GetMaxGFlopDevice();
 
 		checkCudaErrors(cudaSetDevice(deviceID));
 		checkCudaErrors(cudaGetDeviceProperties(&g_deviceProperties, deviceID));
 	}
 
 	int GetMaxGFlopDevice()
 	{
 		int current_device			= 0;
 		int sm_per_multiproc		= 0;
 		int max_compute_perf		= 0;
 		int max_perf_device			= 0;
 		int numCUDAEnabeldDevices	= 0;
 		int best_SM_arch			= 0;
 
 		cudaDeviceProp deviceProp;
 
 		checkCudaErrors(cudaGetDeviceCount(&numCUDAEnabeldDevices));
 
 		//REQUIRES(numCUDAEnabeldDevices > 0 && "NO GPU WITH CUDA SUPPORT!")
 
 		// Find the best major SM Architecture GPU device
 		while (current_device < numCUDAEnabeldDevices)
 		{
 			cudaGetDeviceProperties(&deviceProp, current_device);
 
 			// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
 			if (deviceProp.computeMode != cudaComputeModeProhibited)
 			{
 				if (deviceProp.major > 0 && deviceProp.major < 9999)
 				{
 					best_SM_arch = MAX(best_SM_arch, deviceProp.major);
 				}
 			}
 
 			current_device++;
 		}
 
 			current_device = 0;
 
 		while (current_device < numCUDAEnabeldDevices)
 		{
 			cudaGetDeviceProperties(&deviceProp, current_device);
 
 			// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
 			if (deviceProp.computeMode != cudaComputeModeProhibited)
 			{
 				if (deviceProp.major == 9999 && deviceProp.minor == 9999)
 				{
 					sm_per_multiproc = 1;
 				}
 				else
 				{
 					sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
 				}
 
 				int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
 
 				if (compute_perf  > max_compute_perf)
 				{
 					// If we find GPU with SM major > 2, search only these
 					if (best_SM_arch > 2)
 					{
 						// If our device==dest_SM_arch, choose this, or else pass
 						if (deviceProp.major == best_SM_arch)
 						{
 							max_compute_perf  = compute_perf;
 							max_perf_device   = current_device;
 						}
 					}
 					else
 					{
 						max_compute_perf  = compute_perf;
 						max_perf_device   = current_device;
 					}
 				}
 			}
 
 			++current_device;
 		}
 
 		return max_perf_device;
 	}
}

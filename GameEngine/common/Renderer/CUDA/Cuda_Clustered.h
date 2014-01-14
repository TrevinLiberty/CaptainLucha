// /****************************************************************************/
// /* Copyright (c) 2013, Trevin Liberty
//  * 
//  * Permission is hereby granted, free of charge, to any person obtaining a copy
//  * of this software and associated documentation files (the "Software"), to deal
//  * in the Software without restriction, including without limitation the rights
//  * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  * copies of the Software, and to permit persons to whom the Software is
//  * furnished to do so, subject to the following conditions:
//  * 
//  * The above copyright notice and this permission notice shall be included in
//  * all copies or substantial portions of the Software.
//  * 
//  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  * THE SOFTWARE.
// /****************************************************************************/
// /*
//  *	@author Trevin Liberty
//  *	@file	ClusteredRenderer.h
//  *	@brief	
//  *
// /****************************************************************************/

#ifndef CUDA_CLUSTERASSIGNMENT_H_CL
#define CUDA_CLUSTERASSIGNMENT_H_CL

namespace CaptainLucha
{
	class Cuda_Clustered
	{
	public:
		virtual void Init(
			unsigned int clusterAssignmentBuffer, 
			unsigned int numClustersX, 
			unsigned int numClustersY, 
			unsigned int numClustersZ) = 0;

		virtual void FindUniqueClusters() = 0;
		virtual void CreateLightBVH(int numLights, float* lightPosRadData) = 0;

		virtual unsigned int GetNumUsedClusters() const = 0;
		
		virtual const unsigned int* GetUniqueIndices() const = 0;

		static Cuda_Clustered* CreateRenderer();
	};
}

#endif
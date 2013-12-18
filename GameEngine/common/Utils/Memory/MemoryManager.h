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
 *	@file	MemoryManager.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MEMORY_MANAGER_H_CL
#define MEMORY_MANAGER_H_CL

#include <Threads/fast_mutex.h>
#include <Utils/UtilDebug.h>

typedef unsigned char Byte;

namespace CaptainLucha
{
	struct DataHeader;
	class MemoryManager
	{
	public:
		static MemoryManager& GetInstance()
		{
			static MemoryManager manager;
			return manager;
		}

		void* Allocate(size_t bytes, const char* file, int line);
		void Free(void* data);

		int GetNumFreeAndAllocatedBytes() const;
		int GetNumAllocatedBytes() const {return m_numAllocatedBytes;}
		int GetNumFreeBytes() const;

		void OutputMemoryLeakInfo() const;

		void DisplayVisualizer();
		void DisplayMemoryInfo();

	protected:
		MemoryManager();
		~MemoryManager();

		void* PoolFromTop(size_t bytes, const char* file, int line);
		void* FindLargeEnoughBlockFromTop(size_t bytes);

		void MergeFreeData(Byte* data);

		void RemoveFromFreeList(DataHeader* header);
		void AddToFreeList(DataHeader* header);

	private:
		DataHeader* m_memoryFront;

		DataHeader* m_freeList;

		tthread::fast_mutex m_memoryMutex;

		int m_numAllocations;
		int m_numAllocatedBytes;
		size_t m_largestAllocation;
		int m_averageAllocation;

		int m_currentDebugFrame;
	};
}

void* PoolData(size_t bytes, const char* file, int line);
void FreeData(void* data);

#endif
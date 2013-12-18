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

#include "MemoryManager.h"

#include <Utils/CommonIncludes.h>
#include <Renderer/RendererUtils.h>

#include <iomanip>

void* PoolData(size_t bytes, const char* file, int line)
{
	void* data = CaptainLucha::MemoryManager::GetInstance().Allocate(bytes, file, line);

	if(!data)
	{
		std::bad_alloc e;
		throw e;
	}

	return data;
}

void FreeData(void* data)
{
	CaptainLucha::MemoryManager::GetInstance().Free(data);
}

namespace CaptainLucha
{
	struct DataHeader
	{
		DataHeader* m_prevBlock;
		DataHeader* m_nextBlock;
		//DataHeader* m_nextFree;
		//DataHeader* m_prevFree;
		size_t m_sizeBytes;
		const char* m_fileName;
		int m_line;
		int m_createdFrame;

		DataHeader(DataHeader* prev, DataHeader* next, size_t size, const char* fileName, int lineNumber)
			: m_prevBlock(prev), m_nextBlock(next), m_sizeBytes(size), m_fileName(fileName), m_line(lineNumber) {}
	};

	static int NUM_BYTES_FOR_HEADER = sizeof(DataHeader);

	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	void* MemoryManager::Allocate( size_t bytes, const char* file, int line )
	{
#ifdef MULTITHREADED_APP
		m_memoryMutex.lock();
#endif
		if(bytes > m_largestAllocation)
			m_largestAllocation = bytes;

		m_averageAllocation = (int)(m_averageAllocation * 0.99f + bytes * 0.01f);

		++m_numAllocations;
		void* result = PoolFromTop(bytes, file, line);

		if(result == NULL)
		{
			result;
		}

#ifdef MULTITHREADED_APP
		m_memoryMutex.unlock();
#endif
		return result;
	}

	void MemoryManager::Free( void* data )
	{
		if(data == NULL)
			return;

#ifdef MULTITHREADED_APP
		m_memoryMutex.lock();
#endif
		Byte* headerLoc = ((Byte*)data) - NUM_BYTES_FOR_HEADER;
		DataHeader* temp = reinterpret_cast<DataHeader*>(headerLoc);
		temp->m_line = -1;
		MergeFreeData(headerLoc);

#ifdef MULTITHREADED_APP
		m_memoryMutex.unlock();
#endif
	}

	int MemoryManager::GetNumFreeAndAllocatedBytes() const
	{
		int result = 0;
		DataHeader* header = m_memoryFront;
		while(header != NULL)
		{
			result += header->m_sizeBytes;
			header = header->m_nextBlock;
		}

		return result;
	}

	int MemoryManager::GetNumFreeBytes() const
	{
		int result = 0;
		DataHeader* header = m_memoryFront;
		while(header != NULL)
		{
			if(header->m_line < 0)
				result += header->m_sizeBytes;
			header = header->m_nextBlock;
		}

		return result;
	}

	void MemoryManager::OutputMemoryLeakInfo() const
	{
		DataHeader* currentHeader = reinterpret_cast<DataHeader*>(m_memoryFront);
		while(currentHeader != NULL)
		{
			if(currentHeader->m_line >= 0)
			{
				if(currentHeader->m_fileName != NULL)
					trace(currentHeader->m_fileName << "(" << currentHeader->m_line << ") : MEMORY LEAK! " << currentHeader->m_sizeBytes - NUM_BYTES_FOR_HEADER << " Bytes")
				else
					trace("Unknown(" << currentHeader->m_line << ") : MEMORY LEAK! " << currentHeader->m_sizeBytes - NUM_BYTES_FOR_HEADER << " Bytes")
			}
			currentHeader = currentHeader->m_nextBlock;
		}
	}

	void MemoryManager::DisplayVisualizer()
	{
		++m_currentDebugFrame;
		DataHeader* currentHeader = reinterpret_cast<DataHeader*>(m_memoryFront);

		const float WIDTH = 50.0f;
		float y = WINDOW_HEIGHT;

		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->LoadIdentity();

		SetColor(Color::White);

		DrawBegin(CL_TRIANGLES);
		while(currentHeader != NULL)
		{
			float per = currentHeader->m_sizeBytes / (float)m_numAllocatedBytes;
			float height = WINDOW_HEIGHT * per;
			Color color(0.5f, 0.5f, 1.0f);

			clVertex3((float)WINDOW_WIDTH - WIDTH, y, 1.0f);
			clVertex3((float)WINDOW_WIDTH, y, 1.0f);
			clVertex3((float)WINDOW_WIDTH, y - height, 1.0f);

			clVertex3((float)WINDOW_WIDTH - WIDTH, y, 1.0f);
			clVertex3((float)WINDOW_WIDTH, y - height, 1.0f);
			clVertex3((float)WINDOW_WIDTH - WIDTH, y - height, 1.0f);

			if(currentHeader->m_line < 0)
			{
				float percent = min((m_currentDebugFrame - abs(currentHeader->m_line)) / 150.0f, 1.0f);
				color = Color::White.Interpolate(Color::Red, percent);
			}
			else
			{
				float percent = min((m_currentDebugFrame - abs(currentHeader->m_createdFrame)) / 2500.0f, 1.0f);
				color = Color(0.0f, 0.0f, 0.25f).Interpolate(color, percent);
			}

			clColor4(color);
			clColor4(color);
			clColor4(color);

			clColor4(color);
			clColor4(color);
			clColor4(color);

			y -= height;
			currentHeader = currentHeader->m_nextBlock;
		}

		DrawEnd();

		g_MVPMatrix->Translate(0.0f, 0.0f, 0.9f);
		SetColor(Color::Black);
		DrawLine(
			Vector2Df((float)WINDOW_WIDTH - (float)WIDTH, 0.0f), 
			Vector2Df((float)WINDOW_WIDTH - WIDTH, (float)WINDOW_HEIGHT));

		g_MVPMatrix->PopMatrix();
	}

	void MemoryManager::DisplayMemoryInfo()
	{
		float x = WINDOW_WIDTH * 0.825f;
		float y = WINDOW_HEIGHT - 15.0f;

		SetColor(.2f, .2f, .2f, .75f);
		DrawBegin(CL_QUADS);
		clVertex3(x - 10.0f, (float)WINDOW_HEIGHT, 0.9f);
		clVertex3(x + 250.0f, (float)WINDOW_HEIGHT, 0.9f);
		clVertex3(x + 250.0f, y - 4 * 17.0f, 0.9f);
		clVertex3(x - 10.0f, y - 4 * 17.0f, 0.9f);
		DrawEnd();

		std::stringstream ss;
		ss.precision(5);
		ss << std::setw(10) << std::fixed
			<<	   "Allocated : " << (m_numAllocatedBytes / 1024) / 1024 
			<< "MB\nFree      : " << (GetNumFreeBytes() / 1024) / 1024.0f 
			<< "MB\nUsed      : " << ((GetNumAllocatedBytes() - GetNumFreeBytes()) / 1024) / 1024.0f 
			<< "MB\nAverage Al: " << m_averageAllocation / 1024.0f
			<< "B \nLargest Al: " << (m_largestAllocation / 1024) / 1024
			<< "MB\n";

		Draw2DDebugText(Vector2Df(x, y), ss.str().c_str());
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void* MemoryManager::PoolFromTop(size_t bytes, const char* file, int line)
	{
		const size_t REQUIRED_NUM_BYTES = bytes + NUM_BYTES_FOR_HEADER;
		void* newData = FindLargeEnoughBlockFromTop(REQUIRED_NUM_BYTES);

		if(newData)
		{
			DataHeader* newAllocatedBlock = reinterpret_cast<DataHeader*>(newData);
			DataHeader* nextBlock = newAllocatedBlock->m_nextBlock;
			newAllocatedBlock->m_createdFrame = m_currentDebugFrame;

			if(newAllocatedBlock->m_sizeBytes >= REQUIRED_NUM_BYTES + NUM_BYTES_FOR_HEADER)
			{
				//split
				DataHeader* newFreeBlock = reinterpret_cast<DataHeader*>(((Byte*)newData) + REQUIRED_NUM_BYTES);
				*newFreeBlock = DataHeader(newAllocatedBlock, nextBlock, newAllocatedBlock->m_sizeBytes - REQUIRED_NUM_BYTES, NULL, -1);

				if(newAllocatedBlock->m_nextBlock != NULL)
					newAllocatedBlock->m_nextBlock->m_prevBlock = newFreeBlock;

				nextBlock = newFreeBlock;
				m_freeList = newFreeBlock;
			}

			newAllocatedBlock->m_nextBlock = nextBlock;
			newAllocatedBlock->m_sizeBytes = REQUIRED_NUM_BYTES;
			newAllocatedBlock->m_line = line;
			newAllocatedBlock->m_fileName = file;

			return ((Byte*)newData) + NUM_BYTES_FOR_HEADER;
		}
		else
		{
			//ERROR
		}

		return NULL;
	}

	void* MemoryManager::FindLargeEnoughBlockFromTop(size_t bytes)
	{
		DataHeader* currentHeader = reinterpret_cast<DataHeader*>(m_freeList);
		for(;;)
		{
			if(currentHeader->m_line < 0)
			{
				if(currentHeader->m_sizeBytes == bytes || currentHeader->m_sizeBytes >= bytes + NUM_BYTES_FOR_HEADER)
					return currentHeader;
			}

			currentHeader = currentHeader->m_nextBlock;

			if(currentHeader == NULL)
				currentHeader = m_memoryFront;
			if(currentHeader == currentHeader->m_nextBlock)
				break;
			else if (currentHeader == m_freeList)
				break;
		}
		return NULL;
		//ERROR NO MEMORY AVAILABLE
	}

	void MemoryManager::MergeFreeData(Byte* data)
	{
		DataHeader* header = reinterpret_cast<DataHeader*>(data);
		if(header->m_line < 0)
		{
			m_freeList = header;

			if(header->m_prevBlock != NULL && header->m_prevBlock->m_line < 0)
			{
				DataHeader* prevHeader = header->m_prevBlock;
				prevHeader->m_sizeBytes += header->m_sizeBytes;
				prevHeader->m_nextBlock = header->m_nextBlock;

				m_freeList = prevHeader;

				if(header->m_nextBlock != NULL)
					header->m_nextBlock->m_prevBlock = prevHeader;

				header = prevHeader;
			}

			if(header->m_nextBlock != NULL && header->m_nextBlock->m_line < 0)
			{
				DataHeader* nextHeader = header->m_nextBlock;
				header->m_sizeBytes += nextHeader->m_sizeBytes;
				header->m_nextBlock = nextHeader->m_nextBlock;

				if(nextHeader->m_nextBlock != NULL)
					nextHeader->m_nextBlock->m_prevBlock = header;
			}

			header->m_line = -m_currentDebugFrame;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Private
	//////////////////////////////////////////////////////////////////////////
	MemoryManager::MemoryManager()
		: m_numAllocations(0),
		  m_numAllocatedBytes(0),
		  m_largestAllocation(0),
		  m_averageAllocation(0),
		  m_currentDebugFrame(1)
	{
		m_numAllocatedBytes = (int)((1024 * 1024 * 1024) * 0.5);
		Byte* data = (Byte*)malloc(m_numAllocatedBytes);

		m_memoryFront = reinterpret_cast<DataHeader*>(data);

		*m_memoryFront = DataHeader(NULL, NULL, m_numAllocatedBytes, NULL, -1);

		m_freeList = m_memoryFront;
	}

	MemoryManager::~MemoryManager()
	{

	}
}
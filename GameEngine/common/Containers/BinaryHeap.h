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
 *	@file	BinaryHeap.h
 *	@brief	Templated BinaryHeap tree
 *  @todo	Add support for custom compare function
 *
/****************************************************************************/

#ifndef BINARY_HEAP_H_CL
#define BINARY_HEAP_H_CL

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	template<typename T>
	class BinaryHeap
	{
	public:
		BinaryHeap()
			{m_data.resize(1);}
		~BinaryHeap() {};

		void Reserve(int i) {m_data.reserve(i);}
		bool IsEmpty() {return m_data.size() < 2;}
		void Clear() {m_data.clear(); m_data.resize(1);}
		
		void Push(const T& val);
		void Pop();
		T& Top();
		void RemoveFirstValue(T& val);
		void Sort(int startingPos = 1);
		T* Contains(const T& val);

	private:
		std::vector<T> m_data;

		int GetFirstChild(int i);
		int GetSecondChild(int i);
		int GetParent(int i);
		void SwapNodes(int a, int b);
	};

	template<typename T>
	T& BinaryHeap<T>::Top()
	{
		return m_data[1];
	}

	template<typename T>
	void BinaryHeap<T>::Pop()
	{
		m_data[1] = m_data[m_data.size() - 1];
		m_data.pop_back();
		Sort();
	}

	template<typename T>
	void BinaryHeap<T>::Push( const T& val )
	{
		m_data.push_back(val);

		int i = m_data.size() - 1;
		while (i > 1)
		{
			int parent = GetParent(i);

			if(m_data[i] < m_data[parent])
			{
				SwapNodes(i, parent);
				i = parent;
			}
			else
				break;
		}
	}

	template<typename T>
	int BinaryHeap<T>::GetFirstChild(int i)
	{
		return i * 2;
	}

	template<typename T>
	int BinaryHeap<T>::GetSecondChild(int i)
	{
		return GetFirstChild(i) + 1;
	}

	template<typename T>
	int BinaryHeap<T>::GetParent(int i)
	{
		if(i & 1)
			--i;

		return i  >> 1;
	}

	template<typename T>
	void BinaryHeap<T>::SwapNodes(int a, int b)
	{
		m_data[0] = m_data[a];
		m_data[a] = m_data[b];
		m_data[b] = m_data[0];
	}

	template<typename T>
	void BinaryHeap<T>::RemoveFirstValue(T& val)
	{
		for(size_t i = 1; i < m_data.size(); ++i)
		{
			if(m_data[i] == val)
			{
				m_data[i] = m_data[m_data.size() - 1];
				m_data.pop_back();
				Sort(i);
				break;
			}
		}
	}

	template<typename T>
	void BinaryHeap<T>::Sort(int startingPos)
	{
		size_t i = startingPos;
		while(i < m_data.size())
		{
			size_t left = GetFirstChild(i);
			size_t right = GetSecondChild(i);
			size_t largest = i;

			if(left < m_data.size())
			{
				if(m_data[largest] >= m_data[left])
				{
					largest = left;
				}
			}

			if(right < m_data.size())
			{
				if(m_data[largest] >= m_data[right])
				{
					largest = right;
				}
			}

			if(largest != i)
			{
				SwapNodes(i, largest);
				i = largest;
			}
			else
				break;
		}
	}

	template<typename T>
	T* BinaryHeap<T>::Contains(const T& val)
	{
		for(size_t i = 1; i < m_data.size(); ++i)
		{
			if(m_data[i] == val)
			{
				return &m_data[i];
			}
		}

		return NULL;
	}
}

#endif
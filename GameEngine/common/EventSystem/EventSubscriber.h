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
 *	@file	EventSubscriber.h
 *	@brief	Stores a pointer to the passed object and it function. Used by EventSystem to call object events.
 *  @see	EventSystem
 *
/****************************************************************************/

#ifndef EVENT_SUBSCRIBER_H
#define EVENT_SUBSCRIBER_H

#include "EventSubscriberBase.h"

namespace CaptainLucha
{
	template<typename T>
	class EventSubscriber : public EventSubscriberBase
	{
		typedef void(T::*MemberFunctionType)(NamedProperties&);

	public:
		EventSubscriber(T& object, MemberFunctionType func)
			: object_(&object),
			func_(func) {};
		~EventSubscriber() {};

		virtual void FireEvent(NamedProperties& props)
		{
			(object_->*func_)(props);
		}

	protected:

	private:
		T* object_;
		MemberFunctionType func_;

		friend class EventSystem;
	};
}

#endif
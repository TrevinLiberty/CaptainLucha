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
 *	@file	EventSystem.h
 *	@brief	Used to register individual objects to certain events. Any object can fire events that will then be caught by any other registered events.
 *	@todo	Remove Singleton design.
 *	@see	EventSubscriber NamedProperties
 *
/****************************************************************************/

#ifndef EVENT_SYSTEM_H
#define EVENT_SYSTEM_H

#include "EventSubscriberBase.h"
#include "EventSubscriber.h"
#include "Utils/UtilDebug.h"
#include "Threads/ThreadMutex.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class EventSystem
	{
	public:
		static EventSystem* GetInstance()
		{
			REQUIRES(m_eventSystem)
			return m_eventSystem;
		}

		static void CreateInstance()
		{
			if(!m_eventSystem)
				m_eventSystem = new EventSystem();
		}

		/**
		 * @brief     Fires a event to all subscribed objects. NamedProperties will be empty.
		 */
		void FireEvent(const std::string& event);

		/**
		 * @brief     Fires a event to all subscribed objects.
		 */
		void FireEvent(const std::string& event, NamedProperties& props);

		/**
		 * @brief     Fires a event to all subscribed objects. This templated version creates a named properties using the paramName and param
		 */
		template<typename T>
		void FireEvent(const std::string& event, const std::string& paramName, const T& param)
		{
			NamedProperties np;
			np.Set(paramName, param);
			FireEvent(event, np);
		}

		/**
		 * @brief     Registers an object for the specified event. All fire event calls to that event will now call that object's func
		 */
		template<typename T_Obj, typename T_MemFunc>
		void RegisterObjectForEvent(const std::string& event, T_Obj& obj, const T_MemFunc& func);

		template<typename T_Obj, typename T_MemFunc>
		void UnRegisterObjectForEvent(const std::string& event, const T_Obj& object, const T_MemFunc func);

		template<typename T_Obj>
		void UnRegisterObjectForEvent(const std::string& event, const T_Obj& object);

		template<typename T_Obj>
		void UnRegisterObjectForAllEvents(const T_Obj& object);

		static void DeleteInstance()
		{
			delete m_eventSystem;
			m_eventSystem = NULL;
		}

	protected:
		EventSystem() {};
		~EventSystem();

		template<typename T_Obj>
		void FireSubscriberEvent(EventSubscriberBase* sub);

		template<typename T_Obj, typename T_MemFunc>
		EventSubscriberBase* GetEventSubscriber(const T_Obj& obj, const T_MemFunc& func, std::vector<EventSubscriberBase*>& eventSubs);

	private:
		static EventSystem* m_eventSystem;

		std::map<std::string, std::vector<EventSubscriberBase*> > m_subscribers;
		ThreadMutex m_subscribersMutex;
	};

	void FireEvent(const std::string& event);
	void FireEvent(const std::string& event, NamedProperties& props);

	template<typename T>
	void FireEvent(const std::string& event, const std::string& paramName, const T& param)
	{
		NamedProperties np;
		np.Set(paramName, param);
		FireEvent(event, np);
	}

	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	template<typename T_Obj, typename T_MemFunc>
	void EventSystem::RegisterObjectForEvent(const std::string& event, T_Obj& obj, const T_MemFunc& func)
	{
		ThreadLockGuard<ThreadMutex> guard(m_subscribersMutex);
		std::vector<EventSubscriberBase*>& subs = m_subscribers[event];
		EventSubscriberBase* newSub = new EventSubscriber<T_Obj>(obj, func);

		if(subs.empty())
		{
			subs.push_back(newSub);
		}
		else
		{
			EventSubscriberBase* sub = GetEventSubscriber(obj, func, subs);
			if(sub)
			{
				delete sub;
				sub = newSub;
			}
			else
			{
				subs.push_back(newSub);
			}
		}
	}

	template<typename T_Obj, typename T_MemFunc>
	void EventSystem::UnRegisterObjectForEvent(const std::string& event, const T_Obj& object, const T_MemFunc func)
	{
		ThreadLockGuard<ThreadMutex> guard(m_subscribersMutex);
		std::vector<EventSubscriberBase*>& subs = m_subscribers[event];

		for(auto it = subs.begin(); it != subs.end();)
		{
			EventSubscriber<T_Obj>* sub = dynamic_cast<EventSubscriber<T_Obj>*>((*it));
			if(sub->object_ == &object && sub->func_ == func)
			{
				delete (*it);
				it = subs.erase(it);
			}
			else
				++it;
		}
	}

	template<typename T_Obj>
	void EventSystem::UnRegisterObjectForEvent(const std::string& event, const T_Obj& object)
	{
		ThreadLockGuard<ThreadMutex> guard(m_subscribersMutex);
		std::vector<EventSubscriberBase*>& subs = m_subscribers[event];

		for(auto it = subs.begin(); it != subs.end();)
		{
			EventSubscriber<T_Obj>* sub = dynamic_cast<EventSubscriber<T_Obj>*>((*it));
			if(sub && sub->object_ == &object)
			{
				delete (*it);
				it = subs.erase(it);
			}
			else
				++it;
		}
	}

	template<typename T_Obj>
	void EventSystem::UnRegisterObjectForAllEvents(const T_Obj& object)
	{
		ThreadLockGuard<ThreadMutex> guard(m_subscribersMutex);
		for(auto it = m_subscribers.begin(); it != m_subscribers.end(); ++it)
		{
			std::vector<EventSubscriberBase*>& subs = (*it).second;

			for(auto itt = subs.begin(); itt != subs.end();)
			{
				EventSubscriber<T_Obj>* sub = dynamic_cast<EventSubscriber<T_Obj>*>((*itt));
				if(sub != NULL && sub->object_ == &object)
				{
					delete (*itt);
					itt = subs.erase(itt);
				}
				else
					++itt;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	template<typename T_Obj>
	void EventSystem::FireSubscriberEvent(EventSubscriberBase* sub)
	{
		
	}

	template<typename T_Obj, typename T_MemFunc>
	EventSubscriberBase* EventSystem::GetEventSubscriber(const T_Obj& obj, const T_MemFunc& func, std::vector<EventSubscriberBase*>& eventSubs)
	{
		for(size_t i = 0; i < eventSubs.size(); ++i)
		{
			EventSubscriber<T_Obj>* sub = dynamic_cast<EventSubscriber<T_Obj>*>(eventSubs[i]);
			if(sub && sub->object_ == &obj && sub->func_ == func)
			{
				return sub;
			}
		}

		return NULL;
	}
}

#endif
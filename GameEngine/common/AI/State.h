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
 *	@file	State.h
 *	@brief	
 *
/****************************************************************************/

#ifndef STATE_H_CL
#define STATE_H_CL

#include "Utils/UtilMacros.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	/**
	* @brief	Base class for new states.
	* @see		StateMachine
	*/
	template<typename Actor_Type>
	class State
	{
	public:
		State(const std::string& name) : STATE_NAME(name) {}
		virtual ~State() {}

		/**
		 * @brief     Called when this state become active
		 * @param	  Actor_Type & actor

		 */
		virtual void Enter(Actor_Type& actor) = 0;

		/**
		 * @brief     Called every frame
		 * @param	  Actor_Type & actor

		 */
		virtual void Execute(Actor_Type& actor) = 0;

		/**
		 * @brief     Called when this state is replaced.
		 * @param	  Actor_Type & actor

		 */
		virtual void Exit(Actor_Type& actor) = 0;

		const std::string STATE_NAME;

	private:
		PREVENT_COPYING(State)
	};
}

#endif
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
 *	@file	StateMachine.h
 *	@brief	
 *
/****************************************************************************/

#ifndef STATE_MACHINE_H_CL
#define STATE_MACHINE_H_CL

#include "State.h"
#include "Utils/UtilMacros.h"
#include "Utils/UtilDebug.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	/**
	* @brief     Statemachine for a single object. Holds current, previous, and global states.
		 @code		 
			FooStateMachine stateMachine(Actor);
			stateMachine.InitCurrentState(FooStateA);
			stateMachine.InitGlobalState(FooStateB);
			stateMachine.InitPreviousState(FooStateC);
	
			//Update Loop:
			stateMachine.Update();
		@endcode	 
	*/
	template<typename Actor_Type>
	class StateMachine
	{
	public:
		StateMachine(Actor_Type& owner)
			: owner_(owner),
			  previousState_(NULL),
			  currentState_(NULL),
			  globalState_(NULL)
		{}

		~StateMachine() {};

		/**
		 * @brief     Updates global and current state logic.
		 */
		void Update()
		{
			if (globalState_)
				globalState_->Execute(owner_);

			if (currentState_)
				currentState_->Execute(owner_);
		}

		/**
		 * @brief     Change the current state to the newState. 
		 * This calls the current state's exit function and the nextState's Enter function
		 * @param	  State<Actor_Type> * newState

		 */
		void ChangeState(State<Actor_Type>* newState)
		{
			REQUIRES(newState && "Changing to NULL State")

			previousState_ = currentState_;

			if(currentState_)
				currentState_->Exit(owner_);

			currentState_ = newState;

			currentState_->Enter(owner_);
		}

		/**
		 * @brief     Change state to previous state.

		 */
		void GoToPreviousState() {ChangeState(previousState_);}

		State<Actor_Type>* GetCurrentState() const {return currentState_;}
		State<Actor_Type>* GetPreviousState() const {return previousState_;}
		State<Actor_Type>* GetGlobalState() const {return globalState_;}

		void InitCurrentState(State<Actor_Type>* state) {currentState_ = state;}
		void InitGlobalState(State<Actor_Type>* state) {globalState_ = state;}
		void InitPreviousState(State<Actor_Type>* state) {previousState_ = state;}

	private:
		Actor_Type& owner_;

		State<Actor_Type>* previousState_;
		State<Actor_Type>* currentState_;
		State<Actor_Type>* globalState_;

		PREVENT_COPYING(StateMachine)
	};
}

#endif
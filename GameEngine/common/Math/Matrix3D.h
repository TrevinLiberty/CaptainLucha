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
 *	@file	Matrix3D.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MATRIX3_H_CL
#define MATRIX3_H_CL

#include "Vector3D.h"

#include <math.h>

namespace CaptainLucha
{
	//Row matrix
	template <class Real>
	class Matrix3D
	{
	public:
		Matrix3D(bool isIdentity = false);
		Matrix3D(const Matrix3D& rhs);
		Matrix3D(Real aa, Real ab, Real ac, 
			     Real ba, Real bb, Real bc,
				 Real ca, Real cb, Real cc);
		Matrix3D(const Vector3D<Real>& r1, 
				 const Vector3D<Real>& r2, 
				 const Vector3D<Real>& r3); 

		void MakeZero();
		void MakeIdentity();
		void MakeRotation(const Vector3D<Real>& axis, Real angle);
		void MakeRotation(Real pitch, Real yaw, Real roll = 0);

		Vector3D<Real> GetAxis(int i) const;

		void SetSkewSymmetric(const Vector3D<Real>& vect);

		Real GetValueAtIndex(int i, int j) const;

		Matrix3D GetTranspose() const;
		Matrix3D GetInverse() const;

		Real* Data() {return elements_;}
		const Real* Data() const {return elements_;}

		Matrix3D& operator= (const Matrix3D& rkM);

		bool operator== (const Matrix3D& rhs) const;
		bool operator!= (const Matrix3D& rhs) const;

		Matrix3D operator+ (const Matrix3D& rhs) const;
		Matrix3D operator- (const Matrix3D& rhs) const;
		Matrix3D operator* (const Matrix3D& rhs) const;
		Matrix3D operator* (Real scalar) const;
		Matrix3D operator/ (Real scalar) const;

		Matrix3D& operator+= (const Matrix3D& rhs);
		Matrix3D& operator-= (const Matrix3D& rhs);
		Matrix3D& operator*= (Real fScalar);
		Matrix3D& operator/= (Real fScalar);

		operator Matrix3D<float>() const;
		operator Matrix3D<double>() const;

		Real& operator[] (int i) {return elements_[i];}
		Real operator[] (int i) const {return elements_[i];}

		Vector3D<Real> operator* (const Vector3D<Real>& rhs) const;

		static const Matrix3D ZERO;
		static const Matrix3D IDENTITY;

	private:
		int GetIndex(int i, int j) const;
		int CompareArrays (const Matrix3D& rhs) const;

		Real elements_[9];
	};

typedef Matrix3D<float> Matrix3Df;
typedef Matrix3D<double> Matrix3Dd;

#include "Matrix3D.inl"
}

#endif
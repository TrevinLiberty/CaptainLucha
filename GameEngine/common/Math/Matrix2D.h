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
 *	@file	Matrix2D.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MATRIX2_H_CL
#define MATRIX2_H_CL

#include "Vector2D.h"

namespace CaptainLucha
{
	//Row matrix
	template <class Real>
	class Matrix2D
	{
	public:
		Matrix2D(bool isIdentity = false);
		Matrix2D(const Matrix2D& rhs);
		Matrix2D(Real aa, Real ab,
			     Real ba, Real bb);
		Matrix2D(const Vector2D<Real>& r1, const Vector2D<Real>& r2); 

		void MakeZero();
		void MakeIdentity();

		Vector2D<Real> GetRow(int i);
		Vector2D<Real> GetColumn(int j);

		void SetRow(int row, const Vector2D<Real>& val);
		void SetColumn(int col, const Vector2D<Real>& val);

		Real* GetArray() {return elements_;}

		Real GetValueAtIndex(int i, int j) const;

		Matrix2D GetTranspose() const;
		Matrix2D GetInverse() const;

		Matrix2D& operator= (const Matrix2D& rhs);

		bool operator== (const Matrix2D& rhs) const;
		bool operator!= (const Matrix2D& rhs) const;

		Matrix2D operator+ (const Matrix2D& rhs) const;
		Matrix2D operator- (const Matrix2D& rhs) const;
		Matrix2D operator* (const Matrix2D& rhs) const;
		Matrix2D operator* (Real scalar) const;
		Matrix2D operator/ (Real scalar) const;

		Matrix2D& operator+= (const Matrix2D& rhs);
		Matrix2D& operator-= (const Matrix2D& rhs);
		Matrix2D& operator*= (Real fScalar);
		Matrix2D& operator/= (Real fScalar);

		Real& operator[] (int i) {return elements_[i];}
		Real operator[] (int i) const {return elements_[i];}

		static const Matrix2D ZERO;
		static const Matrix2D IDENTITY;

	private:
		int GetIndex(int i, int j) const;
		int CompareArrays (const Matrix2D& rhs) const;

		Real elements_[4];
	};

typedef Matrix2D<float> Matrix2Df;
typedef Matrix2D<double> Matrix2Dd;

#include "Matrix2D.inl"
}

#endif
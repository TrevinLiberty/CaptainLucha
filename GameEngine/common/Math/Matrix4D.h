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
 *	@file	Matrix4D.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MATRIX4_H_CL
#define MATRIX4_H_CL

#include "Vector4D.h"
#include "Vector3D.h"
#include "Matrix3D.h"

namespace CaptainLucha
{
	class Quaternion;

	//Row matrix
	template <class Real>
	class Matrix4D
	{
	public:
		Matrix4D(bool isIdentity = false);
		Matrix4D(const Matrix4D& rhs);
		Matrix4D(Real radians, Real x, Real y, Real z);
		Matrix4D(Real translateX, Real translateY, Real translateZ);
		Matrix4D(
			Real aa, Real ab, Real ac, Real ad,
			Real ba, Real bb, Real bc, Real bd,
			Real ca, Real cb, Real cc, Real cd,
			Real da, Real db, Real dc, Real dd);


		void MakeZero();
		void MakeIdentity();

		Vector4D<Real> GetRow(int i);
		Vector4D<Real> GetColumn(int j);

		Matrix3D<Real> GetRotation() const;
		Vector3D<Real> GetTranslate() const;

		Vector3D<Real> GetAxis(int i) const;
		
		Matrix4D& MakeRotationRad(Real angle, Real x, Real y, Real z);
		Matrix4D& MakeRotation(Real angle, Real x, Real y, Real z);
		Matrix4D& MakeRotation(const Vector3D<Real>& dir, const Vector3D<Real>& up);
		Matrix4D& MakeTransformation(const Vector3D<Real>& trans, const Quaternion& orientation);

		void SetRotation(const Matrix3D<Real>& rotation);
		void SetTranslate(const Vector3D<Real>& translate);
		void SetSkewSymmetric(const Vector3D<Real>& vect);

		Vector3D<Real> TransformPosition(const Vector3D<Real>& vect) const;
		Vector3D<Real> TransformRotation(const Vector3D<Real>& vect) const;
		Vector4D<Real> Transform(const Vector4D<Real>& vect) const;

		Real* Data() {return elements_;}

		Real GetDeterminant() const;

		Matrix4D GetTranspose() const;
		Matrix4D GetInverse() const;

		void CreateLookAt(const Vector3Df& pos, const Vector3Df& lookAt, const Vector3Df& up);

		Matrix4D& operator= (const Matrix4D& rkM);

		bool operator== (const Matrix4D& rhs) const;
		bool operator!= (const Matrix4D& rhs) const;

		Matrix4D operator+ (const Matrix4D& rhs) const;
		Matrix4D operator- (const Matrix4D& rhs) const;
		Matrix4D operator* (const Matrix4D& rhs) const;
		Matrix4D operator* (Real scalar) const;
		Matrix4D operator/ (Real scalar) const;

		Matrix4D& operator+= (const Matrix4D& rhs);
		Matrix4D& operator-= (const Matrix4D& rhs);
		Matrix4D& operator*= (Real fScalar);
		Matrix4D& operator/= (Real fScalar);

		Vector4D<Real> operator* (const Vector4D<Real>& rhs) const;

		Real& operator[] (int i) {return elements_[i];}
		Real operator[] (int i) const {return elements_[i];}

		operator Matrix4D<float>() const;
		operator Matrix4D<double>() const;

		static const Matrix4D ZERO;
		static const Matrix4D IDENTITY;

	private:
		int GetIndex(int i, int j) const;
		int CompareArrays (const Matrix4D& rhs) const;

		Real elements_[16];
	};

typedef Matrix4D<float> Matrix4Df;
typedef Matrix4D<double> Matrix4Dd;

#include "Matrix4D.inl"
}

#endif
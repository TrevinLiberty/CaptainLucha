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

template<class Real>
Vector4D<Real>& Vector4D<Real>::operator= (const Vector4D& rhs)
{
	x = rhs.x;
	y = rhs.y;
	z = rhs.z;
	w = rhs.w;
	return *this;
}

template<class Real>
bool Vector4D<Real>::operator== (const Vector4D<Real>& rhs) const
{
	return Equals(rhs);
}

template<class Real>
bool Vector4D<Real>::operator!= (const Vector4D<Real>& rhs) const
{
	return !Equals(rhs);
}

template<class Real>
Vector4D<Real> Vector4D<Real>::operator+ (const Vector4D& rhs) const
{
	return Vector4D<Real>(x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w);
}

template<class Real>
Vector4D<Real> Vector4D<Real>::operator- (const Vector4D& rhs) const
{
	return Vector4D<Real>(x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w);
}


template<class Real>
Vector4D<Real> Vector4D<Real>::operator* (Real scalar) const
{
	return Vector4D<Real>(x*scalar, y*scalar, z*scalar, w*scalar);
}

template<class Real>
Vector4D<Real> Vector4D<Real>::operator/ (Real scalar) const
{
	return Vector4D<Real>(x/scalar, y/scalar, z/scalar, w/scalar);
}

template<class Real>
Vector4D<Real>& Vector4D<Real>::operator+= (const Vector4D& rhs)
{
	x += rhs.x;
	y += rhs.y;
	z += rhs.z;
	w += rhs.w;
	return *this;
}

template<class Real>
Vector4D<Real>& Vector4D<Real>::operator-= (const Vector4D& rhs)
{
	x -= rhs.x;
	y -= rhs.y;
	z -= rhs.z;
	w -= rhs.w;
	return *this;
}

template<class Real>
Vector4D<Real>& Vector4D<Real>::operator*= (Real scalar)
{
	x *= scalar;
	y *= scalar;
	z *= scalar;
	w *= scalar;
	return *this;
}

template<class Real>
Vector4D<Real>& Vector4D<Real>::operator/= (Real scalar)
{
	x /= scalar;
	y /= scalar;
	z /= scalar;
	w /= scalar;
	return *this;
}

template<class Real>
Real& Vector4D<Real>::operator[] (int i)
{
	return (&x)[i];
}

template<class Real>
Real Vector4D<Real>::operator[] (int i) const
{
	return (&x)[i];
}

template<class Real>
Real Vector4D<Real>::Length() const
{
	return std::sqrt(SquaredLength());
}

template<class Real>
Real Vector4D<Real>::SquaredLength() const
{
	return this->Dot(*this);
}

template<class Real>
Real Vector4D<Real>::Dot(const Vector4D& rhs) const
{
	return x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w;
}

template<class Real>
void Vector4D<Real>::Normalize()
{
	Real length = Length();

	if(!FloatEquality(length, (Real)0.0))
		*this /= length;
}
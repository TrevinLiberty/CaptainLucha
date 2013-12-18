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
Vector3D<Real>& Vector3D<Real>::operator= (const Vector3D& rhs)
{
	x = rhs.x;
	y = rhs.y;
	z = rhs.z;
	return *this;
}

template<class Real>
bool Vector3D<Real>::operator== (const Vector3D<Real>& rhs) const
{
	return Equals(rhs);
}

template<class Real>
bool Vector3D<Real>::operator!= (const Vector3D<Real>& rhs) const
{
	return !Equals(rhs);
}

template<class Real>
Vector3D<Real> Vector3D<Real>::operator+ (const Vector3D& rhs) const
{
	return Vector3D<Real>(x + rhs.x, y + rhs.y, z + rhs.z);
}

template<class Real>
Vector3D<Real> Vector3D<Real>::operator- (const Vector3D& rhs) const
{
	return Vector3D<Real>(x - rhs.x, y - rhs.y, z - rhs.z);
}

template<class Real>
Vector3D<Real> Vector3D<Real>::operator* (Real scalar) const
{
	return Vector3D<Real>(x*scalar, y*scalar, z*scalar);
}

template<class Real>
Vector3D<Real> Vector3D<Real>::operator/ (Real scalar) const
{
	return Vector3D<Real>(x/scalar, y/scalar, z/scalar);
}

template<class Real>
Vector3D<Real>& Vector3D<Real>::operator+= (const Vector3D& rhs)
{
	x += rhs.x;
	y += rhs.y;
	z += rhs.z;
	return *this;
}

template<class Real>
Vector3D<Real>& Vector3D<Real>::operator-= (const Vector3D& rhs)
{
	x -= rhs.x;
	y -= rhs.y;
	z -= rhs.z;
	return *this;
}

template<class Real>
Vector3D<Real>& Vector3D<Real>::operator*= (Real scalar)
{
	x *= scalar;
	y *= scalar;
	z *= scalar;
	return *this;
}

template<class Real>
Vector3D<Real>& Vector3D<Real>::operator/= (Real scalar)
{
	x /= scalar;
	y /= scalar;
	z /= scalar;
	return *this;
}

template<class Real>
Real& Vector3D<Real>::operator[] (int i)
{
	return (&x)[i];
}

template<class Real>
Real Vector3D<Real>::operator[] (int i) const
{
	return (&x)[i];
}

template<class Real>
Real Vector3D<Real>::Length() const
{
	return std::sqrt(SquaredLength());
}

template<class Real>
Real Vector3D<Real>::SquaredLength() const
{
	return this->Dot(*this);
}

template<class Real>
Real Vector3D<Real>::Dot(const Vector3D& rhs) const
{
	return x * rhs.x + y * rhs.y + z * rhs.z;
}

template<class Real>
void Vector3D<Real>::Normalize()
{
	Real l = Length();

	if(abs(l) > 0.0001)
		*this /= l;
}

template<class Real>
Vector3D<Real> Vector3D<Real>::CrossProduct(const Vector3D& rhs) const
{
	return Vector3D<Real>(
		y * rhs.z - z * rhs.y,
		z * rhs.x - x * rhs.z,
		x * rhs.y - y * rhs.x);
}

template<class Real>
Vector3D<Real>::operator Vector3D<int>() const
{
	return Vector3D<int>
		(static_cast<int>(x), 
		static_cast<int>(y), 
		static_cast<int>(z));
}

template<class Real>
Vector3D<Real>::operator Vector3D<float>() const
{
	return Vector3D<float>(static_cast<float>(x), 
		static_cast<float>(y), 
		static_cast<float>(z));
}

template<class Real>
Vector3D<Real>::operator Vector3D<double>() const
{
	return Vector3D<double>
		(static_cast<double>(x), 
		static_cast<double>(y), 
		static_cast<double>(z));
}
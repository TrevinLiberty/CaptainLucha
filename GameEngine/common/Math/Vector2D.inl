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

template<typename Real>
Vector2D<Real>& Vector2D<Real>::operator= (const Vector2D& rhs)
{
	x = rhs.x;
	y = rhs.y;
	return *this;
}

template<typename Real>
bool Vector2D<Real>::operator== (const Vector2D& rhs) const
{
	return Equals(rhs);
}

template<typename Real>
bool Vector2D<Real>::operator!= (const Vector2D& rhs) const
{
	return !Equals(rhs);
}

template<typename Real>
Vector2D<Real> Vector2D<Real>::operator+ (const Vector2D& rhs) const
{
	return Vector2D<Real>(x + rhs.x, y + rhs.y);
}

template<typename Real>
Vector2D<Real> Vector2D<Real>::operator- (const Vector2D& rhs) const
{
	return Vector2D<Real>(x - rhs.x, y - rhs.y);
}


template<typename Real>
Vector2D<Real> Vector2D<Real>::operator* (Real scalar) const
{
	return Vector2D<Real>(x*scalar, y*scalar);
}

template<typename Real>
Vector2D<Real> Vector2D<Real>::operator/ (Real scalar) const
{
	return Vector2D<Real>(x/scalar, y/scalar);
}

template<typename Real>
Vector2D<Real>& Vector2D<Real>::operator+= (const Vector2D& rhs)
{
	x += rhs.x;
	y += rhs.y;
	return *this;
}

template<typename Real>
Vector2D<Real>& Vector2D<Real>::operator-= (const Vector2D& rhs)
{
	x -= rhs.x;
	y -= rhs.y;
	return *this;
}

template<typename Real>
Vector2D<Real>& Vector2D<Real>::operator*= (Real scalar)
{
	x *= scalar;
	y *= scalar;
	return *this;
}

template<typename Real>
Vector2D<Real>& Vector2D<Real>::operator/= (Real scalar)
{
	x /= scalar;
	y /= scalar;
	return *this;
}

template<typename Real>
float Vector2D<Real>::Length() const
{
	return std::sqrt(static_cast<float>(SquaredLength()));
}

template<typename Real>
Real Vector2D<Real>::SquaredLength() const
{
	return this->Dot(*this);
}

template<typename Real>
Real Vector2D<Real>::Dot(const Vector2D& rhs) const
{
	return x * rhs.x + y * rhs.y;
}

template<typename Real>
void Vector2D<Real>::FastNormalize()
{
	*this *= FastInvSQRT(SquaredLength());
}

template<typename Real>
void Vector2D<Real>::Normalize()
{
	*this /= Length();
}

template<typename Real>
Vector2D<Real> Vector2D<Real>::Reflect(const Vector2D<Real>& norm)
{
	return (norm * this->Dot(norm) * (Real)2.0) - *this;
}

template<typename Real>
Vector2D<Real> Vector2D<Real>::Perp() const
{
	return Vector2D<Real>(-y, x);
}

template<typename Real>
Vector2D<Real> Vector2D<Real>::GetRotate(float radians) const
{
	float c = std::cos(radians);
	float s = std::sin(radians);
	return Vector2D<Real>
		(x * c - y * s,
		y * c + x * s);
}
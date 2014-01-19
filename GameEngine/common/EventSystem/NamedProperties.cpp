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

#include "NamedProperties.h"

namespace CaptainLucha
{
	NamedProperties::NamedProperties(const NamedProperties& np)
	{
		*this = np;
	}

    float NamedProperties::GetFloatParam(int paramIndex)
    {
        std::stringstream ss;
        ss << "param" << paramIndex;

        std::string val;
        Get(ss.str(), val);

        return atof(val.c_str());
    }

	NamedProperties& NamedProperties::operator=(const NamedProperties& np)
	{
		for(auto it = m_properties.begin(); it != m_properties.end(); ++it)
		{
			delete (*it).second;
		}
		m_properties.clear();

		for(auto it = np.m_properties.begin(); it != np.m_properties.end(); ++it)
		{
			m_properties[(*it).first] = (*it).second->CreateCopy();
		}

		return *this;
	}
}
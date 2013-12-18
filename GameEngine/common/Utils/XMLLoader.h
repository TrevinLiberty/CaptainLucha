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
 *	@file	XMLLoader.h
 *	@brief	
 *
/****************************************************************************/

#ifndef XML_LOADER_H_CL
#define XML_LOADER_H_CL

#include "Renderer/Color.h"
#include "Math/Vector2D.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"

#include <Utils/CommonIncludes.h>

#include "pugixml/pugixml.hpp"

namespace CaptainLucha
{
	class XMLNode
	{
	public:
		XMLNode();
		XMLNode(pugi::xml_node node_);
		~XMLNode();

		bool IsValidNode() const;

		XMLNode GetChildWithName(const std::string& name) const;
		XMLNode GetNthChildNodeWithName(const std::string& name, int n) const;
		XMLNode GetFirstChildNode() const;
		XMLNode GetNextSiblingNode() const;

		float GetAttributeAsFloat(const char* attributeName,		float defaultIfNotFound) const;
		int GetAttributeAsInt(const char* attributeName,			int defaultIfNotFound) const;
		bool GetAttributeAsBool(const char* attributeName,			bool defaultIfNotFound) const;
		Color GetAttributeAsColor(const char* attributeName,		const Color& defaultIfNotFound) const;
		std::string GetAttributeAsString(const char* attributeName, const std::string& defaultIfNotFound) const;
		std::string GetPCDataAsString() const;

		Vector2Df GetAttributeAsVector2f(const char* attributeName, const Vector2Df& defaultIfNotFound) const;
		Vector3Df GetAttributeAsVector3f(const char* attributeName, const Vector3Df& defaultIfNotFound) const;
		Vector4Df GetAttributeAsVector4f(const char* attributeName, const Vector4Df& defaultIfNotFound) const;

		Vector2Di GetAttributeAsVector2i(const char* attributeName, const Vector2Di& defaultIfNotFound) const;
		Vector3Di GetAttributeAsVector3i(const char* attributeName, const Vector3Di& defaultIfNotFound) const;
		Vector4Di GetAttributeAsVector4i(const char* attributeName, const Vector4Di& defaultIfNotFound) const;

		std::string GetNodeName() const;

		void ValidateXMLChildElements(const char* commaSeparatedListOfRequiredChildren, const char* commaSeparatedListOfOptionalChildren = "" ) const;
		void ValidateXMLAttributes(const char* commaSeparatedListOfRequiredAttributes, const char* commaSeparatedListOfOptionalAttributes = "" ) const;

		operator bool() const;
		void operator=(const XMLNode& rhs);

	protected:

	private:
		pugi::xml_node node_;

		std::string filePath_;

		friend class XMLLoader;
	};

	class XMLLoader
	{
	public:
		XMLLoader();
		~XMLLoader();

		XMLNode GetRootNode() const {return rootNode_;}

		bool isLoaded() const {return isLoaded_;}

		bool LoadXML(const std::string& filePath);

	protected:

	private:
		std::string filePath_;
		bool isLoaded_;

		pugi::xml_document doc_;
		XMLNode rootNode_;
	};
}

#endif
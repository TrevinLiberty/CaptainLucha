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

#include "XMLLoader.h"
#include "Utils.h"
#include "UtilDebug.h"
#include "../CLCore.h"

#include <iostream>
#include <vector>
#include <sstream>
#include <glew.h>


namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	XMLNode: Public
	//////////////////////////////////////////////////////////////////////////
	XMLNode::XMLNode()
	{

	}

	XMLNode::XMLNode(pugi::xml_node node)
		: node_(node)
	{

	}

	XMLNode::~XMLNode()
	{

	}

	XMLNode XMLNode::GetChildWithName(const std::string& name) const
	{
		if(node_)
		{
			XMLNode node = node_.child(name.c_str());
			node.filePath_ = filePath_;
			return node;
		}
		return *this;
	}

	XMLNode XMLNode::GetNthChildNodeWithName(const std::string& name, int n) const
	{
		REQUIRES(n >= 0)

		if(!node_)
			return *this;

		pugi::xml_node child = node_.child(name.c_str());

		for(int i = 1; i < n; ++i)
		{
			if(!child)
				break;//error, not enough children

			child = child.next_sibling(name.c_str());
		}

		XMLNode node = XMLNode(child);
		node.filePath_ = filePath_;

		return node;
	}

	XMLNode XMLNode::GetFirstChildNode() const
	{
		if(node_)
			return XMLNode(node_.first_child());
		else
			return node_;
	}

	XMLNode XMLNode::GetNextSiblingNode() const
	{
		if(node_)
			return XMLNode(node_.next_sibling());
		else
			return node_;
	}

	float XMLNode::GetAttributeAsFloat(  const char* attributeName, 
		float defaultIfNotFound)  const
	{
		if(node_)
		{
			pugi::xml_attribute attri = node_.attribute(attributeName);

			if(attri)
			{
				return attri.as_float(defaultIfNotFound);
			}
		}
	
		return defaultIfNotFound;
	}

	int XMLNode::GetAttributeAsInt( const char* attributeName, 
		int defaultIfNotFound) const
	{
		if(node_)
		{
			pugi::xml_attribute attri = node_.attribute(attributeName);

			if(attri)
			{
				return attri.as_int(defaultIfNotFound);
			}
		}

		return defaultIfNotFound;
	}

	bool XMLNode::GetAttributeAsBool( const char* attributeName, 
		bool defaultIfNotFound) const
	{
		if(node_)
		{
			pugi::xml_attribute attri = node_.attribute(attributeName);

			if(attri)
			{
				return attri.as_bool(defaultIfNotFound);
			}
		}

		return defaultIfNotFound;
	}

	Color XMLNode::GetAttributeAsColor( const char* attributeName, 
		const Color& defaultIfNotFound) const
	{
		Vector4Di color = GetAttributeAsVector4i(attributeName, Vector4Di(255, 255, 255, 255));

		if(color.x >= 0)
		{
			return Color(color);
		}

		return defaultIfNotFound;
	}

	std::string XMLNode::GetAttributeAsString( const char* attributeName, 
		const std::string& defaultIfNotFound) const
	{
		if(node_)
		{
			pugi::xml_attribute attri = node_.attribute(attributeName);

			if(attri)
			{
				return attri.as_string();
			}
		}

		return defaultIfNotFound;
	}

	std::string XMLNode::GetPCDataAsString() const
	{
		if(node_)
		{
			return node_.child_value();
		}

		return std::string();
	}

	CaptainLucha::Vector2Df XMLNode::GetAttributeAsVector2f(
		const char* attributeName, const Vector2Df& defaultIfNotFound ) const
	{
		std::vector<std::string> values = DelimitString(GetAttributeAsString(attributeName, ""), ',');

		if(values.size() > 1)
		{
			return Vector2Df(
				static_cast<float>(atof(values[0].c_str())), 
				static_cast<float>(atof(values[1].c_str())));
		}

		return defaultIfNotFound;
	}

	CaptainLucha::Vector3Df XMLNode::GetAttributeAsVector3f(  
		const char* attributeName, const Vector3Df& defaultIfNotFound ) const
	{
		std::vector<std::string> values = DelimitString(GetAttributeAsString(attributeName, ""), ',');

		if(values.size() > 2)
		{
			return Vector3Df(
				static_cast<float>(atof(values[0].c_str())), 
				static_cast<float>(atof(values[1].c_str())),
				static_cast<float>(atof(values[2].c_str())));
		}

		return defaultIfNotFound;
	}

	CaptainLucha::Vector4Df XMLNode::GetAttributeAsVector4f( 
		const char* attributeName, const Vector4Df& defaultIfNotFound ) const
	{
		std::vector<std::string> values = DelimitString(GetAttributeAsString(attributeName, ""), ',');

		if(values.size() > 3)
		{
			return Vector4Df(
				static_cast<float>(atof(values[0].c_str())), 
				static_cast<float>(atof(values[1].c_str())),
				static_cast<float>(atof(values[2].c_str())), 
				static_cast<float>(atof(values[3].c_str())));
		}

		return defaultIfNotFound;
	}

	CaptainLucha::Vector2Di XMLNode::GetAttributeAsVector2i( 
		const char* attributeName, const Vector2Di& defaultIfNotFound ) const
	{
		std::vector<std::string> values = DelimitString(GetAttributeAsString(attributeName, ""), ',');

		if(values.size() > 1)
		{
			return Vector2Di(
				atoi(values[0].c_str()), 
				atoi(values[1].c_str()));
		}

		return defaultIfNotFound;
	}

	CaptainLucha::Vector3Di XMLNode::GetAttributeAsVector3i( 
		const char* attributeName, const Vector3Di& defaultIfNotFound ) const
	{
		std::vector<std::string> values = DelimitString(GetAttributeAsString(attributeName, ""), ',');

		if(values.size() > 2)
		{
			return Vector3Di(
				atoi(values[0].c_str()), 
				atoi(values[1].c_str()),
				atoi(values[2].c_str()));
		}

		return defaultIfNotFound;
	}

	CaptainLucha::Vector4Di XMLNode::GetAttributeAsVector4i(  
		const char* attributeName, const Vector4Di& defaultIfNotFound ) const
	{
		std::vector<std::string> values = DelimitString(GetAttributeAsString(attributeName, ""), ',');

		if(values.size() > 3)
		{
			return Vector4Di(
				atoi(values[0].c_str()), 
				atoi(values[1].c_str()),
				atoi(values[2].c_str()), 
				atoi(values[3].c_str()));
		}

		return defaultIfNotFound;
	}

	std::string XMLNode::GetNodeName() const
	{
		if(node_)
		{
			return node_.name();
		}
		else
			return "";
	}

	void XMLNode::ValidateXMLChildElements(const char* commaSeparatedListOfRequiredChildren, 
		const char* commaSeparatedListOfOptionalChildren) const
	{
		std::vector<std::string> reqTokens = DelimitString(commaSeparatedListOfRequiredChildren, ',');
		std::vector<std::string> optTokens = DelimitString(commaSeparatedListOfOptionalChildren, ',');

		bool error = false;
		std::string errors;

		for(size_t i = 0; i < reqTokens.size(); ++i)
		{
			if(!node_.child(reqTokens[i].c_str()))
			{
				error = true;
				errors.append("XML ERROR:\n\tREQUIRED child: " + reqTokens[i] + " doesn't exsist in node: " + node_.name() + "\n");
				std::string dirPath = GetCurrentDirectoryPath_w() + "\\" + filePath_ + "(0)";
				traceWithFile(dirPath);
			}
		}

		if(error)
		{
			error = false;
			OutputErrorMessage(errors);
		}

		pugi::xml_node child = node_.first_child();

		while(child)
		{
			bool foundAttr = false;
			for(size_t i = 0; i < reqTokens.size(); ++i)
			{
				if(_strcmpi(child.name(), reqTokens[i].c_str()) == 0)
				{
					foundAttr = true;
					break;
				}
			}

			for(size_t j = 0; j < optTokens.size() && !foundAttr; ++j)
			{
				if(_strcmpi(child.name(), optTokens[j].c_str()) == 0)
				{
					foundAttr = true;
					break;
				}
			}

			if(!foundAttr)
			{
				error = true;
				std::string attName = child.name();
				errors.append("XML ERROR:\tChild Node Not Recognized: " + attName + "\n");
				std::string dirPath = GetCurrentDirectoryPath_w() + "\\" + filePath_ + "(0)";
				traceWithFile(dirPath);
			}

			child = child.next_sibling();
		}

		if(error)
		{
			OutputErrorMessage(errors);
		}
	}
	
	void XMLNode::ValidateXMLAttributes(const char* commaSeparatedListOfRequiredAttributes,
		const char* commaSeparatedListOfOptionalAttributes) const
	{
		std::vector<std::string> reqTokens = DelimitString(commaSeparatedListOfRequiredAttributes, ',');
		std::vector<std::string> optTokens = DelimitString(commaSeparatedListOfOptionalAttributes, ',');

		bool error = false;
		std::string errors;

		for(size_t i = 0; i < reqTokens.size(); ++i)
		{
			if(!node_.attribute(reqTokens[i].c_str()))
			{
				error = true;
				errors.append("XML ERROR:\n\tREQUIRED child: " + reqTokens[i] + " doesn't exsist in node: " + node_.name() + "\n");
				std::string dirPath = GetCurrentDirectoryPath_w() + "\\" + filePath_ + "(0)";
				traceWithFile(dirPath);
			}
		}

		if(error)
		{
			error = false;
			OutputErrorMessage(errors);
		}

		pugi::xml_attribute_iterator att = node_.attributes_begin();

		for(att; att != node_.attributes_end(); ++att)
		{
			bool foundAttr = false;
			for(size_t i = 0; i < reqTokens.size(); ++i)
			{
				if(_strcmpi(att->name(), reqTokens[i].c_str()) == 0)
				{
					foundAttr = true;
					break;
				}
			}

			for(size_t j = 0; j < optTokens.size() && !foundAttr; ++j)
			{
				if(_strcmpi(att->name(), optTokens[j].c_str()) == 0)
				{
					foundAttr = true;
					break;
				}
			}

			if(!foundAttr)
			{
				error = true;
				std::string attName = att->name();
				errors.append("XML ERROR:\tAttribute Not Recognized: " + attName + "\n");
				std::string dirPath = GetCurrentDirectoryPath_w() + "\\" + filePath_ + "(0)";
				traceWithFile(dirPath);
			}
		}

		if(error)
		{
			OutputErrorMessage(errors);
		}
	}

	XMLNode::operator bool() const
	{
		if(node_)
			return true;
		else
			return false;
	}

	void XMLNode::operator=(const XMLNode& rhs)
	{
		node_ = rhs.node_;
		filePath_ = rhs.filePath_;
	}

	//////////////////////////////////////////////////////////////////////////
	//	XML LOADER: Public
	//////////////////////////////////////////////////////////////////////////
	XMLLoader::XMLLoader()
		: isLoaded_(false)
	{

	}

	XMLLoader::~XMLLoader()
	{

	}

	bool XMLLoader::LoadXML( const std::string& filePath )
	{
		pugi::xml_parse_result result = doc_.load_file(filePath.c_str());

		if(result)
		{
			rootNode_ = doc_.first_child();
			isLoaded_ = true;
			filePath_ = filePath;
			rootNode_.filePath_ = filePath_;

			return true;
		}
		else
		{
			isLoaded_ = false;

			int lineNumber = GetLineNumberFromByteOffset(filePath, result.offset);

			std::stringstream ss;
			ss << "File: " + filePath + "\nLine Number: " << lineNumber << "\n\nDescription:\n" << result.description();
			std::string msgBoxError = ss.str();
			ss.str("");

			std::string dirPath = GetCurrentDirectoryPath_w() + "\\" + filePath + "(";
			ss << dirPath << lineNumber << ")";

			//GLFW glutHideWindow();
			traceWithFile(ss.str());
			MessageBox(NULL, msgBoxError.c_str(),"XML ERROR", MB_OK | MB_ICONERROR);
			//GLFW glutShowWindow();

			DebugBreak();

			return false;
		}
	}
}
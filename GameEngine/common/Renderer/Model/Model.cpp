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

#include "Model.h"

#include "Utils/CLFileImporter.h"
#include "Renderer/RendererUtils.h"
#include "Renderer/Texture/GLTextureManager.h"
#include "Renderer/Shader/GLProgram.h"
#include "Model_Node_Mesh.h"
#include "Renderer/Geometry/Mesh.h"

namespace CaptainLucha
{
	Model::Model(CLCore& clCore, const char* filePath)
		: m_clCore(clCore),
		  m_numNodes(0),
		  m_filePath(filePath)
	{
		CLFileImporter fileImporter;
		CLFile* file = fileImporter.LoadFile(filePath);

		std::stringstream ss;
		ss.write(file->m_buffer, file->m_bufLength);
		ss >> std::skipws;

		delete file;

		LoadMaterials(ss);
		LoadNodes(ss);

		CalculateAABB();
	}

	Model::~Model()
	{
		trace(m_baseTranslation)
		std::fstream file(m_filePath);
		file.seekp(-3 * 4, std::ios_base::end);
		file.write((char*)&m_baseTranslation, 3 * 4);
	}

	void Model::Draw(GLProgram& glProgram)
	{
		glProgram.SetUniform("emissive", 0.0);

		g_MVPMatrix->PushMatrix();
		g_MVPMatrix->Translate(m_baseTranslation);

		if(m_root)
			m_root->Draw(glProgram, this);
		g_MVPMatrix->PopMatrix();
	}

	void Model::LoadMaterials(std::stringstream& ss)
	{
		std::string word;

		ss >> word; 
		m_name = word;

		ss >> word;
		bool isSkinned = atoi(word.c_str()) != 0;
		UNUSED(isSkinned)

		ss >> word;
		ss >> word;
		int numMaterials = atoi(word.c_str());

		m_materials.resize(numMaterials);

		for(int i = 0; i < numMaterials; ++i)
		{
			ss >> word; ss >> word;

			ss >> word; ss >> word;
			m_materials[i].Diffuse(LoadTexture(word));
			ss >> word; ss >> word;
			m_materials[i].Specular(LoadTexture(word));
			ss >> word; ss >> word;
			m_materials[i].Emissive(LoadTexture(word));
			ss >> word; ss >> word;
			m_materials[i].Normal(LoadTexture(word));
			ss >> word; ss >> word;
			m_materials[i].Mask(LoadTexture(word));

			ss >> word;
			m_materials[i].DiffuseColor(LoadColor(ss));
			ss >> word; 
			m_materials[i].SpecularColor(LoadColor(ss));
			ss >> word; 
			m_materials[i].EmissiveColor(LoadColor(ss));

			ss >> word;
			m_materials[i].Opacity(LoadFloat(ss));
			ss >> word;
			m_materials[i].SpecIntensity(LoadFloat(ss));
		}
	}

	Model_Node* CreateTree(std::vector<std::pair<Model_Node*, int> >& nodes)
	{
		Model_Node* rootNode = nodes[0].first;

		std::stack<std::pair<Model_Node*, int>> nodeStack;
		nodeStack.push(nodes[0]);

		for(size_t i = 1; i < nodes.size(); ++i)
		{
			std::pair<Model_Node*, int>& node = nodes[i];

			if(node.second == nodeStack.top().second + 1)
			{
				node.first->SetParent(nodeStack.top().first);
				nodeStack.push(node);
			}
			else
			{
				for(;;)
				{
					nodeStack.pop();
					if(nodeStack.size() == 1 || node.second == nodeStack.top().second + 1)
					{
						node.first->SetParent(nodeStack.top().first);
						nodeStack.push(node);
						break;
					}
				}
			}
		}

		return rootNode;
	}

	void Model::LoadNodes(std::stringstream& ss)
	{
		std::string word;
		int numNodes;

		ss >> word;
		ss >> numNodes;

		m_numNodes = numNodes;

		std::vector<std::pair<Model_Node*, int> > nodes;

		for(int i = 0; i < numNodes; ++i)
		{
			ss >> word;
			ss >> word;
			RemoveQuotes(word);

			std::string name = word;
			int numMeshs;
			ss >> numMeshs;

			int treeDepth;
			ss >> treeDepth;

			Matrix4Df trans;
			ss.read((char*)&trans, 16 * 4);

			if(numMeshs > 0)
			{
				Model_Node_Mesh* meshNode = new Model_Node_Mesh(name, trans);
				nodes.push_back(std::pair<Model_Node*, int>(
					meshNode, treeDepth));

				std::vector<std::pair<Mesh*, int> > nodeMeshs;
				LoadMesh(ss, numMeshs, nodeMeshs);

				for(size_t i = 0; i < nodeMeshs.size(); ++i)
					meshNode->AddMesh(nodeMeshs[i].first, nodeMeshs[i].second);
			}
			else
			{
				nodes.push_back(std::pair<Model_Node*, int>(
					new Model_Node(name, trans), treeDepth));
			}
		}

		m_root = CreateTree(nodes);

		ss.read((char*)&m_baseTranslation, 3 * 4);
		trace(m_baseTranslation)
	}

	void Model::LoadMesh(std::stringstream& ss, int numMeshs, std::vector<std::pair<Mesh*, int> >& outMeshs)
	{
		std::string word;
		char temp;
		int materialIndex, numVerts, numIndices;

		for(int i = 0; i < numMeshs; ++i)
		{
			ss >> word;
			ss >> materialIndex;
			ss >> numVerts;
			ss >> numIndices;
			
			ss >> temp;

			std::vector<TangentSpaceVertex> verts;
			std::vector<unsigned int> indices;

			verts.resize(numVerts);
			indices.resize(numIndices);

			//ss >> std::skipws;
			ss.read((char*)verts.data(), numVerts * sizeof(TangentSpaceVertex));
			ss.read((char*)indices.data(), numIndices * sizeof(int));
			//ss >> std::skipws;

			Mesh* newMesh = new Mesh(verts, indices);
			outMeshs.push_back(std::pair<Mesh*, int>(newMesh, materialIndex));
		}
	}

	GLTexture* Model::LoadTexture(const std::string& path)
	{
		if(strcmp("na", path.c_str()) != 0)
		{
			std::stringstream ss;
			ss << "Data\\Models\\" << path;
			return m_clCore.GetLevelTextures().CreateOrGetTexture(ss.str(), true);
		}
		else
			return NULL;
	}

	Color Model::LoadColor(std::stringstream& ss)
	{
		Color result;
		ss >> result.r;
		ss >> result.g;
		ss >> result.b;
		result.a = 1.0f;
		return result;
	}

	float Model::LoadFloat(std::stringstream& ss)
	{
		float result;
		ss >> result;
		return result;
	}

	void Model::RemoveQuotes(std::string& quotedName)
	{
		if(!quotedName.empty())
			quotedName = quotedName.substr(1, quotedName.size() - 2);
	}

	void Model::CalculateAABB()
	{
		m_root->GetAABB(m_AABB);
		m_AABB.Transform(m_baseTranslation);
	}
}
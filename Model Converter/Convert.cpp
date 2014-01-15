#include "Convert.h"
#include "WindowsUtils.h"

#include <iostream>
#include <Windows.h>
#include <sstream>
#include <fstream>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/config.h>
#include <assimp/types.h>

namespace CaptainLucha
{
	void ExportMaterials(const aiScene* SCENE, const char* fullPath, const char* exportPath, std::ofstream& file);
	void ExportNodeData(const aiScene* SCENE, int numNodes, std::ofstream& file);

	void ExportFloat(std::ofstream& file, const char* title, float val);
	void ExportColor(std::ofstream& file, const char* title, const aiColor3D& color);
	void ExportTexture(std::ofstream& file, const char* title, const aiString& path, const char* importPath, const char* exportPath);

	int GetNumberOfNodes(const aiScene* SCENE);

	void GetFileNameFromPath(const std::string& path, std::string& outName)
	{
		char temp[MAX_PATH];
		char fileName[MAX_PATH];
		char ext[MAX_PATH];
		_splitpath_s(path.c_str(), temp, temp, fileName, ext);

		outName = fileName;
		outName.append(ext);
	}

	void GetFilePathWithoutFile(const std::string& fullPath, std::string& outPath)
	{
		char temp[MAX_PATH];
		char fileName[MAX_PATH];
		_splitpath_s(fullPath.c_str(), temp, fileName, temp, temp);

		outPath = fileName;
	}

	void OutputError(const char* errorMsg)
	{
		MessageBox(NULL, errorMsg, NULL, MB_OK);
		std::cout << errorMsg << "\n\n\n";
	}

	void ReplaceSpaces(std::string& str)
	{
		for(size_t i = 0; i < str.size(); ++i)
			if(str[i] == ' ')
				str[i] = '_';
	}

	void ConvertAndExportFile(const char* filePath, const char* exportDirectory)
	{
		char unused[MAX_PATH];
		char fullPath[MAX_PATH];
		char name[MAX_PATH];
		_splitpath_s(filePath, unused, fullPath, name, unused);

		std::string fileName = name;

		Assimp::Importer importer;
		importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, 
			aiComponent_COLORS |
			aiComponent_CAMERAS |
			aiComponent_LIGHTS |
			aiComponent_NORMALS |
			aiComponent_TANGENTS_AND_BITANGENTS);
		importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE,
			aiPrimitiveType_POINT | aiPrimitiveType_LINE);
		importer.SetPropertyInteger(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 80);

		//These settings optimize the mesh tree into one node. To support skinning, the tree must still exsist. 
		//	todo skinning.
		const aiScene* SCENE = importer.ReadFile(filePath, 
			aiProcess_CalcTangentSpace |
			aiProcess_JoinIdenticalVertices |
			aiProcess_Triangulate |
			aiProcess_RemoveComponent |
			aiProcess_GenSmoothNormals |
			aiProcess_ValidateDataStructure |
			aiProcess_SortByPType |
			aiProcess_FindDegenerates |
			aiProcess_FindInvalidData |
			aiProcess_GenUVCoords |
			aiProcess_OptimizeMeshes |
			aiProcess_OptimizeGraph 
			);

		if(!SCENE)
		{
			std::cout << importer.GetErrorString();
			OutputError(importer.GetErrorString());
			return;
		}

		if(!SCENE->HasMeshes())
		{
			std::string error = filePath;
			error += "\n File contains no Meshes";
			OutputError(error.c_str());
			return;
		}

		ReplaceSpaces(fileName);
		std::string exportFileName = exportDirectory;
		std::string directoryName = fileName;

		directoryName[0] = (char)toupper(directoryName[0]);
		exportFileName.append(directoryName + "/");
		CreateDirectory(exportFileName.c_str(), NULL);

		std::string textureDirectory = exportFileName;
		textureDirectory.append("Textures/");
		CreateDirectory(textureDirectory.c_str(), NULL);

		exportFileName.append(fileName);
		exportFileName.append(".lm");
		std::ofstream outfile(exportFileName, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
		int numNodes = GetNumberOfNodes(SCENE);

		outfile << fileName << " " << SCENE->HasAnimations() << "\n";
		ExportMaterials(SCENE, fullPath, directoryName.c_str(), outfile);
		ExportNodeData(SCENE, numNodes, outfile);

		float zero[3] = {0.0f, 0.0f, 0.0f};
		outfile.write((char*)&zero, 3 * 4);

		outfile.close();
		importer.FreeScene();
		return;
	}

	void ExportMaterials(const aiScene* SCENE, const char* fullPath, const char* exportPath, std::ofstream& file)
	{
		file << "Materials " << SCENE->mNumMaterials << "\n";
		for(unsigned int i = 0; i < SCENE->mNumMaterials; ++i)
		{
			aiMaterial& material = *SCENE->mMaterials[i];
			std::string name;

			aiString diffusePath;
			aiString specularPath;
			aiString emissivePath;
			aiString normalPath;
			aiString maskPath;
			aiColor3D diffuseColor;
			aiColor3D specularColor;
			aiColor3D emissiveColor;
			float opacity;
			float shininess;

			material.Get(AI_MATKEY_NAME, name);
			material.GetTexture(aiTextureType_DIFFUSE, 0, &diffusePath);
			material.GetTexture(aiTextureType_SPECULAR, 0, &specularPath);
			material.GetTexture(aiTextureType_EMISSIVE, 0, &emissivePath);
			material.GetTexture(aiTextureType_HEIGHT, 0, &normalPath);		//Bug with assimp. Sometimes normal map is in height or bump map slots. todo FIX
			material.GetTexture(aiTextureType_OPACITY, 0, &maskPath);

			material.Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor);
			material.Get(AI_MATKEY_COLOR_SPECULAR, specularColor);
			material.Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor);
			material.Get(AI_MATKEY_OPACITY, opacity);
			material.Get(AI_MATKEY_SHININESS, shininess);

			file << "M " << i << " " << name << "\n";
			ExportTexture(file, "dM ", diffusePath, fullPath, exportPath);
			ExportTexture(file, "sM ", specularPath, fullPath, exportPath);
			ExportTexture(file, "eM ", emissivePath, fullPath, exportPath);
			ExportTexture(file, "nM ", normalPath, fullPath, exportPath);
			ExportTexture(file, "mM ", maskPath, fullPath, exportPath);
			ExportColor(file, "dC ", diffuseColor);
			ExportColor(file, "sC ", specularColor);
			ExportColor(file, "eC ", emissiveColor);
			ExportFloat(file, "op ", opacity);
			ExportFloat(file, "sh ", shininess);

			file << "\n";
		}
	}

	struct Mesh
	{
		int numVerts;
		int numIndices;
		int materialIndex;
	};

	struct Vert
	{
		aiVector3D v;
		aiVector3D n;
		aiVector2D uv;
		aiVector3D t;
		aiVector3D b;

		Vert(const Vert& rhs)
		{
			memcpy(this, &rhs, sizeof(Vert));
		}

		Vert(
			const aiVector3D& v, 
			const aiVector3D& n, 
			const aiVector3D& b, 
			const aiVector3D& t, 
			const aiVector2D& uv)
			: v(v), n(n), b(b), t(t), uv(uv)
		{}
	};

	aiVector3D operator*(const aiVector3D& lhs, const aiMatrix4x4& rhs)
	{
		aiVector3D result;

		result.x = lhs.x * rhs.a1 + lhs.y * rhs.a2 + lhs.z * rhs.a3 + rhs.a4;
		result.y = lhs.x * rhs.b1 + lhs.y * rhs.b2 + lhs.z * rhs.b3 + rhs.b4;
		result.z = lhs.x * rhs.c1 + lhs.y * rhs.c2 + lhs.z * rhs.c3 + rhs.c4;

		return result;
	}

	//If texture cords exist, return them.
	inline aiVector2D GetUVCoords(const aiVector3D* uv, int index)
	{
		if(uv)
			return aiVector2D(uv[index].x, uv[index].y);
		else
			return aiVector2D();
	}

	static int ttt = 0;
	void ExportNode(const aiScene* SCENE, const aiNode* node, std::ofstream& file, aiMatrix4x4 accTransform, int treeDepth)
	{
		file << "N \"" << node->mName.C_Str() << "\" " << node->mNumMeshes << " " << treeDepth;
		std::cout << "N \"" << ++ttt << " "  << node->mName.C_Str() << "\" " << node->mNumMeshes << " " << treeDepth;	

		//Write worldspace Transformation for current node
		file.write((char*)&(accTransform.Transpose()), 16 * 4);

		std::vector<Vert> verts;
		std::vector<int> indices;

		std::cout << std::endl;
		for(size_t i = 0; i < node->mNumMeshes; ++i)
		{
			aiMesh& mesh = *SCENE->mMeshes[node->mMeshes[i]];
			
			for(size_t vi = 0; vi < mesh.mNumVertices; ++vi)
			{
				verts.push_back(Vert(
					mesh.mVertices[vi],
					mesh.mNormals[vi],
					mesh.mBitangents[vi],
					mesh.mTangents[vi],
					mesh.mTextureCoords ? GetUVCoords(mesh.mTextureCoords[0], vi) : aiVector2D()));
			}

			for(size_t ini = 0; ini != mesh.mNumFaces; ++ini)
			{
				aiFace& face = mesh.mFaces[ini];

				if(face.mNumIndices == 3)
				{
					indices.push_back(face.mIndices[0]);
					indices.push_back(face.mIndices[1]);
					indices.push_back(face.mIndices[2]);
				}
			}

			std::cout << "\tMesh " << i << std::endl;
			std::cout << "\t\tnv: " << verts.size() << std::endl;
			std::cout << "\t\tni: " << indices.size() << std::endl << std::endl;

			file << "M " << mesh.mMaterialIndex << " " << verts.size() << " " << indices.size() << " V";

			file.write((char*)&verts[0], verts.size() * sizeof(Vert));
			file.write((char*)&indices[0], indices.size() * sizeof(int));

			verts.clear();
			indices.clear();
		}
	}

	void ParseNodeChildren(const aiScene* SCENE, const aiNode* node, std::ofstream& file, aiMatrix4x4 accTransform, int treeDepth = 0)
	{
		aiMatrix4x4 newTransform = node->mTransformation * accTransform;
		ExportNode(SCENE, node, file, newTransform, treeDepth);

		for(size_t i = 0; i < node->mNumChildren; ++i)
			ParseNodeChildren(SCENE, node->mChildren[i], file, newTransform, treeDepth + 1);
	}

	void ExportNodeData(const aiScene* SCENE, int numNodes, std::ofstream& file)
	{
		file << "NodeData " << numNodes << "\n";
		std::cout << std::endl;

		ParseNodeChildren(SCENE, SCENE->mRootNode, file, aiMatrix4x4());
	}

	void ExportFloat(std::ofstream& file, const char* title, float val)
	{
		file << title << val << "\n";
	}

	void ExportColor(std::ofstream& file, const char* title, const aiColor3D& color)
	{
		file << title << color.r << " " << color.g << " " << color.b << "\n";
	}

	//Hacky. Some textures, mainly normal map, would have -bm 0.0200 prepended to the path.
	//	This specifically removes that from the path, leaving you with only the path to the texture.
	//	Assimp adds this to the path.
	void RemoveAnyModifiers(std::string& path)
	{
		std::stringstream ss(path);
		char c;
		ss >> c;

		if(c == '-')
		{
			std::string word;
			float temp;
			ss >> word;
			ss >> temp;
			ss >> std::skipws;
			getline(ss, path);
			path = path.substr(1);
		}
	}

	//replaces all spaces with underscores
	void ReplaceAnySpaces(std::string& path)
	{
		std::stringstream ss(path);
		path.clear();
		char c;

		ss >> std::noskipws;
		while(ss >> c)
		{
			if(c == ' ')
				path.push_back('_');
			else
				path.push_back(c);
		}
	}

	//Copies the texture at path (if the texture exists) to a new folder "Textures"
	//	Also writes to file a new line. "title""relativePath" or "title"na if the texture doesn't exist.
	void ExportTexture(std::ofstream& file, const char* title, const aiString& path, const char* importPath, const char* exportPath)
	{
		std::string texturePath = path.C_Str();
		RemoveAnyModifiers(texturePath);
		std::string texturePathWithoutSpaces = texturePath;
		ReplaceAnySpaces(texturePathWithoutSpaces);

		std::string textureName;
		GetFileNameFromPath(texturePathWithoutSpaces, textureName);

		std::string relativeExpPath = exportPath;
		relativeExpPath.append("/Textures/");
		relativeExpPath.append(textureName);

		if(path.length > 0)
		{
			file << title << relativeExpPath.c_str() << "\n";

			std::string actualImportPath = importPath;
			actualImportPath.append(texturePath);

			std::string actualExportPath = "Data/Export/";
			actualExportPath.append(relativeExpPath);

			CopyFile(actualImportPath.c_str(), actualExportPath.c_str(), false);
		}
		else
			file << title << "na" << "\n";
 	}

	int GetNumberOfNodesP(const aiScene* SCENE, const aiNode* node)
	{
		int numNodes = 1;
		for(size_t i = 0; i < node->mNumChildren; ++i)
			numNodes += GetNumberOfNodesP(SCENE, node->mChildren[i]);

		return numNodes;
	}

	int GetNumberOfNodes(const aiScene* SCENE)
	{
		return GetNumberOfNodesP(SCENE, SCENE->mRootNode);
	}
}
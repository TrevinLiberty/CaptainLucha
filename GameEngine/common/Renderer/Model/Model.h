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
 *	@file	Model.h
 *	@brief	
 *
/****************************************************************************/

#ifndef MODEL_H_CL
#define MODEL_H_CL

#include "CLCore.h"
#include "Renderer/Renderable.h"
#include "Renderer/Model/Model_Material.h"
#include "Renderer/Color.h"
#include "Renderer/RendererUtils.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	class Model_Node;
	class GLProgram;
	class GLTexture;
	class Mesh;

	class Model : public Renderable
	{
	public:
		Model(CLCore& clCore, const char* filePath);
		~Model();

		void Draw(GLProgram& glProgram, Renderer& renderer);

		int GetNumMaterials() const {return m_materials.size();}
		int GetNumNodes() const {return m_numNodes;}

		const std::string& GetFilePath() const {return m_filePath;}
		void SetFilePath(const std::string& val) {m_filePath = val;}

		const Vector3Df& GetBaseTranslation() const {return m_baseTranslation;}
		void SetBaseTranslation(const Vector3Df& val) {m_baseTranslation = val;}

        void SetScale(float scale) {m_scale = scale;}

		const std::string& GetName() const {return m_name;}

	protected:
		void LoadMaterials(std::stringstream& ss);
		void LoadNodes(std::stringstream& ss);
		void LoadMesh(std::stringstream& ss, int numMeshs, std::vector<std::pair<Mesh*, int> >& outMeshs);

		GLTexture* LoadTexture(const std::string& path);
		Color LoadColor(std::stringstream& ss);
		float LoadFloat(std::stringstream& ss);

		void RemoveQuotes(std::string& word);
		
	private:
		Model_Node* m_root;

		Vector3Df m_baseTranslation;
        float m_scale;

		std::vector<Model_Material> m_materials;
		int m_numNodes;

		CLCore& m_clCore;

		std::string m_filePath;
		std::string m_name;

		PREVENT_COPYING(Model)

		friend class Model_Node_Mesh;
	};
}

#endif
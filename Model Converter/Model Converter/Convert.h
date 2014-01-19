#ifndef CONVERT_H_CL
#define CONVERT_H_CL

namespace CaptainLucha
{
	//Converts the model file at filePath in a .lm model file and exports it to
    //  exportDirectory. This will create a new folder in exportDirectory named
    //  using the original file's name. The folder will contain a sub folder 
    //  that will have all the needed textures for the model.
	//		All texture path's start from the newly create model folder.
	//
	//Attention. This assumes that filePath is directed to a model that assimp
    //              supports.
	//
	//	Example .lm file
	//	Everything in quotes represent a variable that can change. Note, 
    //      NodeName is in double quotes.
	//	// is a comment and is not represented in the .lm file
	//
	//	ModelName "NonZero if animations are present, else 0"
	//	Materials "NumberOfMaterials"
	//	M "MatNumber"
	//	dM Sponza\Textures\sponza_thorn_diff.tga //Diffuse  Color Map Filepath
	//	sM Sponza\Textures\sponza_thorn_spec.tga //Specular Map Filepath
	//	eM na									 //Emissive Map Filepath
	//	nM na									 //Normal Map Filepath
	//	mM na									 //Opacity Map Filepath
	//	dC 0.58 0.58 0.58						 //Diffuse Color
	//	sC 0 0 0								 //Specular Color
	//	eC 0 0 0								 //Emissive Color
	//	op 1									 //Opacity
	//	sh 40									 //Shininess (SpecularExponent)
	//
	//	...ect for more materials
	//
	//	NodeData "NodeNumber"
	//	N ""NodeName"" "NumMeshs" "treeDepth"
	//	"4x4 worldTransformation in binary format"
	//	M "MaterialIndex" "NumberOfVerts" "NumberOfIndices"
	//	"Vert Data followed by Indices data in binary format"
	//
	//	...ect for more nodes
	void ConvertAndExportFile(
        const char* filePath, 
        const char* exportDirectory);
}

#endif
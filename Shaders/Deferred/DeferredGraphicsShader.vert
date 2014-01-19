#version 330

/////////////////////////////////////////////////////
//Uniforms
uniform mat4x4 model;
uniform mat4x4 modelViewProj;

/////////////////////////////////////////////////////
//Ins
in vec3 vertex;
in vec3 normal;
in vec2 texCoord;
in vec3 tangent;
in vec3 bitangent;

/////////////////////////////////////////////////////
//Outs
out vec3 worldPosition;
out vec3 vertNormal;
out vec2 textureCoord;
out vec3 vertTangent;
out vec3 vertbiTangent;

void main()
{
	gl_Position		= modelViewProj * vec4(vertex, 1.0);

	worldPosition	= (model * vec4(vertex, 1.0)).xyz;

	vertNormal		= normal;
	vertTangent		= tangent;
	vertbiTangent	= bitangent;

	textureCoord	= texCoord;
}

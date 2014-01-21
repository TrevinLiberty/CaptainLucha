#version 330

uniform mat4x4 modelViewProj;

uniform float pointSize;

in vec3 vertex;
in vec4 vertexColor;
in vec2 texCoord;

out vec2 textureCoord;
out vec4 vertColor;

void main()
{
	gl_Position = modelViewProj * vec4(vertex, 1.0);

	textureCoord = texCoord;
	vertColor = vertexColor; 

	gl_PointSize = 3.0;
}

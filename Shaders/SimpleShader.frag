#version 330

uniform bool useTexture;
uniform bool useVertColor;

uniform sampler2D diffuseMap;
uniform vec4 color;

in vec2 textureCoord;
in vec4 vertColor;

out vec4 FragColor;

void main()
{
	FragColor = color;
}
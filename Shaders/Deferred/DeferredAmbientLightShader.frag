#version 330

uniform sampler2D renderTarget0;
uniform sampler2D renderTarget1;
uniform sampler2D renderTarget2;

uniform vec4 color;
uniform float intensity;

/////////////////////////////////////////////////////
//Ins
in vec2 textureCoord;

layout(location=0) out vec3 DiffuseTarget;
layout(location=1) out vec3 SpecularTarget;

void main()
{
	DiffuseTarget = color.rgb * intensity;
	SpecularTarget = vec3(0, 0, 0);
}
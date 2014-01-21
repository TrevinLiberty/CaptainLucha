#version 330
 
uniform mat4x4 modelViewProj;

in vec3 vertex;

void main()
{
	gl_Position = modelViewProj * vec4(vertex, 1.0);
}
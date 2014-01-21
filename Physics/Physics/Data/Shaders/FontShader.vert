#version 330

uniform mat4x4 modelViewProj;

in vec3 vertex;
in vec2 tex_coord;
in vec4 color;

out vec2 textureCoord;
out vec4 fColor;

void main()
{
	gl_Position = modelViewProj * vec4(vertex, 1.0);
	textureCoord = tex_coord;
	fColor = color;
}

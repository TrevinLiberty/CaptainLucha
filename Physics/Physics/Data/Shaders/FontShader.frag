#version 330

uniform sampler2D diffuseMap;

in vec2 textureCoord;
in vec4 fColor;

out vec4 FragColor;

void main()
{
	FragColor = vec4(fColor.rgb, fColor.a * texture(diffuseMap, textureCoord).r);
}
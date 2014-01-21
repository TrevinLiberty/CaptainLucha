#version 330

uniform sampler2D depthTexture;

in vec2 textureCoord;
in vec4 vertColor;

out vec4 FragColor;

void main()
{
	float depth = texture(depthTexture, textureCoord).r;

	depth = pow(depth, 128);

	FragColor = vec4(depth, depth, depth, 1.0);
}
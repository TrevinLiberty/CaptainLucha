#version 430

/////////////////////////////////////////////////////
//Uniforms
uniform sampler2D renderTarget0;
uniform sampler2D renderTarget1;

uniform sampler2D accumulatorDiffuse;
uniform sampler2D accumulatorSpecular;

/////////////////////////////////////////////////////
//Ins
in vec2 textureCoord;

/////////////////////////////////////////////////////
//Outs
out vec4 FragColor;

void main()
{
	vec3 lightDiffuse	= texture(accumulatorDiffuse, textureCoord).rgb;
	vec3 lightSpecular	= texture(accumulatorSpecular, textureCoord).rgb;

	vec4 diffuse		= texture(renderTarget0, textureCoord);
	float emissive		= texture(renderTarget1, textureCoord).a;

	FragColor.rgb		= 
			(lightDiffuse * diffuse.rgb)
			+ lightSpecular
			+ emissive * diffuse.rgb;

	FragColor.a = 1.0;
}
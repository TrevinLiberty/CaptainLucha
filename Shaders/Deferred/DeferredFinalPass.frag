#version 430

uniform sampler2D renderTarget0;
uniform sampler2D renderTarget1;

uniform sampler2D accumulatorDiffuse;
uniform sampler2D accumulatorSpecular;

in vec2 textureCoord;

out vec4 FragColor;

void main()
{
	vec3 lightDiffuse  = texture(accumulatorDiffuse, textureCoord).rgb;
	vec3 lightSpecular = texture(accumulatorSpecular, textureCoord).rgb;

	vec3 diffuse   = texture(renderTarget0, textureCoord).rgb;
	float emissive = texture(renderTarget1, textureCoord).a;

	lightDiffuse = clamp(lightDiffuse, 0.0, 1.0);

	FragColor.rgb = (lightDiffuse * diffuse) + lightSpecular + emissive * diffuse;
	FragColor.a = 1.0;
}
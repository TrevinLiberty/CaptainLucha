#version 330

uniform sampler2D renderTarget0;
uniform sampler2D renderTarget1;

uniform vec4  color;
uniform float intensity;
uniform vec3  lightDir;

/////////////////////////////////////////////////////
//Ins
in vec2 textureCoord;

layout(location=0) out vec3 DiffuseTarget;
layout(location=1) out vec3 SpecularTarget;

void main()
{
	vec3 texelColor = texture(renderTarget0, textureCoord).rgb;
	vec3 fragNormal = texture(renderTarget1, textureCoord).xyz;
	
	float normalDot = max(dot(fragNormal, -lightDir), 0.0); 

	DiffuseTarget = texelColor * color.rgb * intensity * normalDot;
	SpecularTarget = vec3(0, 0, 0);
}
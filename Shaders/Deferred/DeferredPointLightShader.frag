#version 330

uniform sampler2D renderTarget0;
uniform sampler2D renderTarget1;
uniform sampler2D renderTarget2;

uniform vec4 color;

uniform vec3 camPos;
uniform vec3 lightPos;

uniform float intensity;
uniform float radius;

/////////////////////////////////////////////////////
//Out
layout(location=0) out vec3 DiffuseTarget;
layout(location=1) out vec3 SpecularTarget;

float CalculateNormalDot(in vec3 fragNormal, in vec3 dirToLight)
{
	return max(dot(fragNormal, dirToLight), 0.0); 
}

float CalculateAttenuation(in float dist)
{
	dist = dist / radius;
	return clamp(1.0 - (dist * dist), 0.0, 1.0);
}

float CalculateSpecular(in vec3 fragNormal, in vec3 viewDir, in vec3 dirToLight, in int specExponent)
{
	vec3 reflection = reflect(dirToLight, fragNormal);
	return pow(max(0.0, dot(reflection, viewDir)), specExponent);
}

void main()
{
	vec2 textureCoord	= gl_FragCoord.xy / vec2(1620, 920);
	float spec			= texture(renderTarget0, textureCoord).a;
	vec3 fragNormal		= texture(renderTarget1, textureCoord).xyz;
	vec4 worldPos		= texture(renderTarget2, textureCoord).xyzw;
	int specExponent    = int(worldPos.w);
	vec3 viewDir		= normalize(worldPos.xyz - camPos);

	vec3 dirToLight		= lightPos - worldPos.xyz;
	float distToLight	= length(dirToLight);
	dirToLight			/= distToLight;

	float normalDot		= max(CalculateNormalDot(fragNormal, dirToLight), 0.0);
	float attenuation	= clamp(CalculateAttenuation(distToLight), 0.0, 1.0);
	float specular		= clamp(CalculateSpecular(fragNormal, viewDir, dirToLight, specExponent), 0.0, 1.0);
	
	DiffuseTarget = color.rgb * attenuation * intensity * normalDot;
	SpecularTarget = color.rgb * attenuation * intensity * specular * normalDot * spec;
}
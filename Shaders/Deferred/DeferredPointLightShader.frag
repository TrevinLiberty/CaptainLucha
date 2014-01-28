#version 330

//Todo: make uniforms
const float INV_SCREEN_W = 1 / 1620.0;
const float INV_SCREEN_H = 1 / 920.0;

/////////////////////////////////////////////////////
//Uniforms
uniform sampler2D renderTarget0;
uniform sampler2D renderTarget1;
uniform sampler2D renderTarget2;

uniform vec4 color;

uniform vec3 camPos;
uniform vec3 lightPos;

uniform float intensity;
uniform float radius;

uniform mat4x4 view;
uniform mat4x4 projection;

/////////////////////////////////////////////////////
//Ins

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

vec3 GetWorldPositionFromDepth(float depth)
{
	vec4 ndc = vec4(gl_FragCoord.xy, depth, 1.0);
	ndc.x = (ndc.x * INV_SCREEN_W) * 2 - 1;
	ndc.y = (ndc.y * INV_SCREEN_H) * 2 - 1;
	ndc.z = ndc.z * 2 - 1;

	ndc = inverse(projection) * ndc;
	ndc.xyz /= ndc.w;

	return (inverse(view) * vec4(ndc.xyz, 1.0)).xyz;
}

void main()
{
	vec2 textureCoord	= gl_FragCoord.xy * vec2(INV_SCREEN_W, INV_SCREEN_H);

	vec3 fragNormal		= texture(renderTarget1, textureCoord).xyz;
	vec4 renderTarget2	= texture(renderTarget2, textureCoord).xyzw;

	int specExponent	= int(renderTarget2.w);
	float spec			= renderTarget2.y;

	vec3 worldPos		= GetWorldPositionFromDepth(renderTarget2.x);
	vec3 viewDir		= normalize(worldPos - camPos);

	vec3 dirToLight		= lightPos - worldPos;
	float distToLight	= length(dirToLight);
	dirToLight			/= distToLight;

	float normalDot		= max(CalculateNormalDot(fragNormal, dirToLight), 0.0);
	float attenuation	= clamp(CalculateAttenuation(distToLight), 0.0, 1.0);
	float specular		= clamp(CalculateSpecular(fragNormal, viewDir, dirToLight, specExponent), 0.0, 1.0);
	
	DiffuseTarget = color.rgb * attenuation * intensity * normalDot;

	if(dot(dirToLight, fragNormal) > 0.0)
		SpecularTarget = color.rgb * attenuation * intensity * specular * normalDot * spec;
	else
		SpecularTarget = vec3(0.0);
}
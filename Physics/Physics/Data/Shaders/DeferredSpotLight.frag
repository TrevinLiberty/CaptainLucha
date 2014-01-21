#version 330

uniform sampler2D renderTarget0;
uniform sampler2D renderTarget1;
uniform sampler2D renderTarget2;

uniform vec4 color;

uniform vec3 camPos;
uniform vec3 lightPos;
uniform vec3 lightDir;

uniform float intensity;
uniform float radius;

uniform float innerAngle;
uniform float outerAngle;

/////////////////////////////////////////////////////
//Out
layout(location=0) out vec3 DiffuseTarget;
layout(location=1) out vec3 SpecularTarget;

float CalculateNormalDot(vec3 fragNormal)
{
	return max(dot(fragNormal, -lightDir), 0.0); 
}

float CalculateSpotlight(vec3 dirToLight)
{
	float spotLight = dot(dirToLight, -lightDir);
	return smoothstep(outerAngle, innerAngle, spotLight);
}

float CalculateAttenuation(float distToLight)
{
	float atts = 1;
	float att = (distToLight*distToLight)/(radius*radius);
	att = 1 / (att*atts+1);

	atts = 1 / (atts + 1);
	att = att - atts;

	return att / (1 - atts);
}

float CalculateSpecular(vec3 fragNormal, vec3 viewDir)
{
	const int SPECULAR_EXPONENT = 32;
	vec3 reflection = reflect(-lightDir, fragNormal);
	return pow(max(0.0, dot(reflection, viewDir)), SPECULAR_EXPONENT);
}

void main()
{
	vec2 textureCoord = gl_FragCoord.xy / vec2(1620, 920);
	vec3 fragNormal = texture(renderTarget1, textureCoord).xyz;
	vec3 worldPos = texture(renderTarget2, textureCoord).xyz;
	vec3 viewDir = normalize(worldPos - camPos);

	vec3 dirToLight = lightPos - worldPos;
	float distToLight = length(dirToLight);
	dirToLight /= distToLight;

	float normalDot = CalculateNormalDot(fragNormal);
	float spotLight = CalculateSpotlight(dirToLight);
	float attenuation = CalculateAttenuation(distToLight);
	float specular = CalculateSpecular(fragNormal, viewDir);

	DiffuseTarget = color.rgb * intensity * attenuation * spotLight * normalDot;
	SpecularTarget = vec3(0, 0, 0);
}
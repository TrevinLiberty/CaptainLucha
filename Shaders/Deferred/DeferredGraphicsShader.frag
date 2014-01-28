#version 330

/////////////////////////////////////////////////////
//Uniforms
uniform sampler2D diffuseMap;
uniform sampler2D normalMap;
uniform sampler2D specularMap;
uniform sampler2D emissiveMap;
uniform sampler2D maskMap;

uniform bool hasDiffuseMap;
uniform bool hasNormalMap;
uniform bool hasSpecularMap;
uniform bool hasEmissiveMap;
uniform bool hasMaskMap;

uniform float emissive;
uniform int specularIntensity;

uniform vec4 color;

/////////////////////////////////////////////////////
//Ins
in vec3 worldPosition;
in vec3 vertNormal;
in vec2 textureCoord;
in vec3 vertTangent;
in vec3 vertbiTangent;

/////////////////////////////////////////////////////
//Outs
layout(location=0) out vec4 RT0;
layout(location=1) out vec4 RT1;
layout(location=2) out vec4 RT2;

void main()
{
	vec4 texelColor = color;

	if(hasDiffuseMap)//non-divergent
		texelColor *= texture(diffuseMap, textureCoord);

	if(hasMaskMap && texture(maskMap, textureCoord).r < 0.25)//divergent
		discard;

	//Normal Mapping
	vec3 normal = normalize(vertNormal);
	if(hasNormalMap)//non-divergent
	{
		mat3 tsmatrix = mat3(vertTangent, vertbiTangent, normal);
		normal = (2 * texture(normalMap, textureCoord).xyz) - 1;
		normal = normalize(tsmatrix * normal);
	}

	//Specular
	float specVal = 0.1f;
	if(hasSpecularMap)//non-divergent
		specVal = texture(specularMap, textureCoord).r;
	
	//Emissive
	float emissiveVal = emissive;
	if(hasEmissiveMap)//non-divergent
		emissiveVal = texture(emissiveMap, textureCoord).r;

	//Render Target out projection
	RT0 = vec4(texelColor.rgb, 1.0);
	RT1 = vec4(normal, emissiveVal);
	RT2 = vec4(gl_FragCoord.z, specVal, 0.0, specularIntensity);
}

//	modelposition

//modelmatrix
//	world space

//viematrix
//	view space

//projection
//	clip space

//w divide
//	ndc	[-1, 1]

//scaled
//	window coords

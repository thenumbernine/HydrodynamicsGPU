varying vec2 texCoord;

#ifdef VERTEX_SHADER

void main() {
	texCoord = gl_MultiTexCoord0.xy;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}

#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER

uniform bool useLog;
//1/log(10)
#define _1_LN_10 0.4342944819032517611567811854911269620060920715332

uniform float scale;
uniform sampler2D tex;
uniform sampler1D gradient;
uniform float alpha;

void main() {
	float value = texture2D(tex, texCoord).r;
	if (useLog) value = log(1. + abs(value)) * _1_LN_10;
	value *= scale;
	gl_FragColor = vec4(texture1D(gradient, value).rgb, alpha);
}

#endif	//FRAGMENT_SHADER

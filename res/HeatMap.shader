varying vec2 texCoord;

#ifdef VERTEX_SHADER

void main() {
	texCoord = gl_MultiTexCoord0.xy;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}

#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER

uniform float scale;
uniform sampler2D tex;
uniform sampler1D gradient;

void main() {
	float value = texture2D(tex, texCoord).r;
	gl_FragColor = texture1D(gradient, value * scale);
}

#endif	//FRAGMENT_SHADER


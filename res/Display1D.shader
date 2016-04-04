varying vec3 color;

#ifdef VERTEX_SHADER

uniform sampler2D tex;
uniform float xmin;
uniform float xmax;
uniform float scale;

void main() {
	vec3 vertex = gl_Vertex.xyz;
	vertex.y = texture2D(tex, vertex.xy).r * scale;
	vertex.x = vertex.x * (xmax - xmin) + xmin;
	color = gl_Color.rgb;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(vertex, 1.);
}

#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER

void main() {
	gl_FragColor = vec4(color, 1.);
}

#endif	//FRAGMENT_SHADER

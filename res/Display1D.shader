#ifdef VERTEX_SHADER

uniform float xmin, xmax;
uniform sampler2D tex;

void main() {
	vec3 vertex = gl_Vertex.xyz;
	vertex.y = texture2D(tex, vertex.xy).r;
	vertex.x = vertex.x * (xmax - xmin) + xmin;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(vertex, 1.);
}

#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER

void main() {
	gl_FragColor = vec4(1., 1., 1., 1.);
}

#endif	//FRAGMENT_SHADER


varying vec3 color;

#ifdef VERTEX_SHADER

uniform sampler2D tex;
uniform int axis;
uniform vec2 xmin;
uniform vec2 xmax;
uniform float scale;

void main() {
	vec3 vertex = gl_Vertex.xyz;
	vertex[axis] = texture2D(tex, vertex.xy).r * scale;
	vertex.x = vertex.x * (xmax.x - xmin.x) + xmin.x;
	vertex.y = vertex.y * (xmax.y - xmin.y) + xmin.y;
	color = gl_Color.rgb;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(vertex, 1.);
}

#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER

void main() {
	gl_FragColor = vec4(color, 1.);
}

#endif	//FRAGMENT_SHADER



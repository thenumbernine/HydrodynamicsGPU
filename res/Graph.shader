varying vec3 color;
varying vec3 normal;

#ifdef VERTEX_SHADER

uniform sampler2D tex;
uniform int axis;
uniform vec2 xmin;
uniform vec2 xmax;
uniform float scale;
uniform bool useLog;

vec3 func(vec3 src) {
	vec3 vertex = src.xyz;
	vertex.x = vertex.x * (xmax.x - xmin.x) + xmin.x;
	vertex.y = vertex.y * (xmax.y - xmin.y) + xmin.y;
	vertex[axis] = texture2D(tex, src.xy).r * scale;
	if (useLog) vertex[axis] = log(1. + abs(vertex[axis]));
	return vertex;
}

void main() {
	vec3 vertex = func(gl_Vertex.xyz);

	const float epsilon = 1./256.;	//TODO 1/gridsize
	vec3 xp = func(gl_Vertex.xyz + vec3(epsilon, 0., 0.));
	vec3 xm = func(gl_Vertex.xyz - vec3(epsilon, 0., 0.));
	vec3 yp = func(gl_Vertex.xyz + vec3(0., epsilon, 0.));
	vec3 ym = func(gl_Vertex.xyz - vec3(0., epsilon, 0.));

	normal = normalize(cross(xp - xm, yp - ym));

	color = gl_Color.rgb;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(vertex, 1.);
}

#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER

uniform float ambient;

void main() {
	vec3 light = normalize(vec3(.5, .5, 1.));
	float lum = dot(normal, light);
	//lum = max(lum, -lum);	//two-sided
	lum = max(lum, ambient);
	gl_FragColor = vec4(color * lum, 1.);
}

#endif	//FRAGMENT_SHADER

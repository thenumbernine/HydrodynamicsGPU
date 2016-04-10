varying vec4 color;
varying vec3 normal;

#ifdef VERTEX_SHADER

uniform sampler2D tex;
uniform int axis;
uniform vec2 xmin;
uniform vec2 xmax;
uniform float scale;
uniform bool useLog;
uniform vec2 size;

//1/log(10)
#define _1_LN_10 0.4342944819032517611567811854911269620060920715332

vec3 func(vec3 src) {
	vec3 vertex = src.xyz;
	vertex.x = vertex.x * (xmax.x - xmin.x) + xmin.x;
	vertex.y = vertex.y * (xmax.y - xmin.y) + xmin.y;
	vertex[axis] = texture2D(tex, src.xy).r * scale;
	if (useLog) vertex[axis] = log(1. + abs(vertex[axis])) * _1_LN_10;
	return vertex;
}

void main() {
	vec3 vertex = func(gl_Vertex.xyz);

	vec3 xp = func(gl_Vertex.xyz + vec3(1./size.x, 0., 0.));
	vec3 xm = func(gl_Vertex.xyz - vec3(1./size.x, 0., 0.));
	vec3 yp = func(gl_Vertex.xyz + vec3(0., 1./size.y, 0.));
	vec3 ym = func(gl_Vertex.xyz - vec3(0., 1./size.y, 0.));

	normal = normalize(cross(xp - xm, yp - ym));

	color = gl_Color.rgba;
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
	gl_FragColor = vec4(color.rgb * lum, color.a);
}

#endif	//FRAGMENT_SHADER

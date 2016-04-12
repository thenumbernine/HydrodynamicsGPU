varying vec3 texCoordStart;
varying vec3 vertexStart;
varying vec3 eye;

#ifdef VERTEX_SHADER

void main() {
	//this only needs to be done once per render
	//i.e. uniform?
	eye = (gl_ModelViewMatrixInverse * vec4(0., 0., 0., 1.)).xyz;
	texCoordStart = gl_MultiTexCoord0.xyz;
	vertexStart = gl_Vertex.xyz;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}

#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER

uniform sampler3D tex;
uniform sampler1D gradient;
uniform int maxiter;
uniform vec3 oneOverDx;
uniform float scale;
uniform bool useLog;
//1/log(10)
#define _1_LN_10 0.4342944819032517611567811854911269620060920715332

float getValue(vec3 p) {
	float value = texture2D(tex, p).r;
	if (useLog) value = log(1. + abs(value)) * _1_LN_10;
	value *= scale;
	return value;
}

void main() {
	float alpha;
	vec3 p = texCoordStart;
	vec4 result = vec4(0., 0., 0., 1.);
	
	float value = getValue(p); 
	
	alpha = .7 * min(1., mod(value * 4., 3.));
	result.rgb += result.a * alpha * texture1D(gradient, value);
	result.a *= 1. - alpha;
	
	vec3 step = vertexStart - eye;
	step = normalize(step) / float(maxiter);
	step /= oneOverDx;
	for (int i = 2; i <= maxiter; i++) {
		p += step;
		if (p.x < 0. || p.y < 0. || p.z < 0. ||
			p.x > 1. || p.y > 1. || p.z > 1.) break;

		//notice if you store 1-alpha in the color you trace through the volume
		//then you can forward-trace your blending
		//instead of having to backward-trace
		//(as you would when rendering transparent stuff on top of each other)
		//this will allow you to bailout early if your transparency ever hits fully opaque
		value = texture3D(tex, p);
		alpha = .7 * min(1., mod(value * 4., 3.));
		result.rgb += result.a * alpha * texture1D(gradient, value);
		result.a *= 1. - alpha;

		if (result.a < .01) break;
	}
	gl_FragColor = vec4(result.rgb, 1. - result.a);
}

#endif	//FRAGMENT_SHADER


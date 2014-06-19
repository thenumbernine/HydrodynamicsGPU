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
uniform int maxiter;
uniform vec3 scale;

void main() {
	float alpha;
	vec3 p = texCoordStart;
	vec4 result = vec4(0., 0., 0., 1.);
	
	vec4 voxel = texture3D(tex, p);
	alpha = .7 * min(1., mod(voxel.r * 1.23, 3.));
	result.rgb += result.a * alpha * voxel.rgb;
	result.a *= 1. - alpha;
	
	vec3 step = vertexStart - eye;
	step = normalize(step) * 2. / float(maxiter);
	step /= scale;
	for (int i = 2; i <= maxiter; i++) {
		p += step;
		if (p.x < 0. || p.y < 0. || p.z < 0. ||
			p.x > 1. || p.y > 1. || p.z > 1.) break;

		//notice if you store 1-alpha in the color you trace through the volume
		//then you can forward-trace your blending
		//instead of having to backward-trace
		//(as you would when rendering transparent stuff on top of each other)
		//this will allow you to bailout early if your transparency ever hits fully opaque
		voxel = texture3D(tex, p);
		alpha = .7 * min(1., mod(voxel.r * 1.23, 3.));
		result.rgb += result.a * alpha * voxel.rgb;
		result.a *= 1. - alpha;

		if (result.a < .01) break;
	}
	gl_FragColor = vec4(result.rgb, 1. - result.a);
}

#endif	//FRAGMENT_SHADER


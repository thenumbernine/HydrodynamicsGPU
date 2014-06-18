varying vec3 v;
varying vec3 eye;

#ifdef VERTEX_SHADER

void main() {
	//this only needs to be done once per render
	//i.e. uniform?
	eye = (gl_ModelViewMatrixInverse * vec4(0., 0., 0., 1.)).xyz;
	v = gl_Vertex.xyz;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}

#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER

uniform sampler3D tex;
uniform int maxiter;

void main() {
	float alpha;
	vec3 p = v;
	vec4 result = vec4(0., 0., 0., 1.);
	
	vec4 voxel = texture3D(tex, p + .5);
	alpha = .7 * min(1., mod(voxel.r * 1.23, 3.));
	result.rgb += result.a * alpha * voxel.rgb;
	result.a *= 1. - alpha;
	
	vec3 step = v - eye;
	step = normalize(step) * 2. / float(maxiter);
	for (int i = 2; i <= maxiter; i++) {
		p += step;
		if (p.x < -.5 || p.y < -.5 || p.z < -.5 ||
			p.x > .5 || p.y > .5 || p.z > .5) break;

		//notice if you store 1-alpha in the color you trace through the volume
		//then you can forward-trace your blending
		//instead of having to backward-trace
		//(as you would when rendering transparent stuff on top of each other)
		//this will allow you to bailout early if your transparency ever hits fully opaque
		voxel = texture3D(tex, p + .5);
		alpha = .7 * min(1., mod(voxel.r * 1.23, 3.));
		result.rgb += result.a * alpha * voxel.rgb;
		result.a *= 1. - alpha;

		if (result.a < .01) break;
	}
	gl_FragColor = vec4(result.rgb, 1. - result.a);
}

#endif	//FRAGMENT_SHADER


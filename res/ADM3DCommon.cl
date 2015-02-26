#include "HydroGPU/Shared/Common.h"

float sym33determinant(const __global real* m) {
	return m[SYM33_XX] * m[SYM33_YY] * m[SYM33_ZZ]
		+ m[SYM33_XY] * m[SYM33_YZ] * m[SYM33_XZ]
		+ m[SYM33_XZ] * m[SYM33_XY] * m[SYM33_YZ]
		- m[SYM33_XZ] * m[SYM33_YY] * m[SYM33_XZ]
		- m[SYM33_YZ] * m[SYM33_YZ] * m[SYM33_XX]
		- m[SYM33_ZZ] * m[SYM33_XY] * m[SYM33_XY];
}

//using Cramer's method
void sym33inverse(const __global real* m, real* mInv) {
	float det = sym33determinant(m);
	mInv[SYM33_XX] = (m[SYM33_YY] * m[SYM33_ZZ] - m[SYM33_YZ] * m[SYM33_YZ]) / det;
	mInv[SYM33_YY] = (m[SYM33_XX] * m[SYM33_ZZ] - m[SYM33_XZ] * m[SYM33_XZ]) / det;
	mInv[SYM33_ZZ] = (m[SYM33_XX] * m[SYM33_YY] - m[SYM33_XY] * m[SYM33_XY]) / det;
	mInv[SYM33_XY] = (m[SYM33_XY] * m[SYM33_ZZ] - m[SYM33_YZ] * m[SYM33_XZ]) / det;
	mInv[SYM33_XZ] = (m[SYM33_XY] * m[SYM33_YZ] - m[SYM33_YY] * m[SYM33_XZ]) / det;
	mInv[SYM33_YZ] = (m[SYM33_XX] * m[SYM33_YZ] - m[SYM33_XY] * m[SYM33_XZ]) / det;
}

//specific to Euler equations
__kernel void convertToTex(
	const __global real* stateBuffer,
	__write_only image3d_t fluidTex,
	__read_only image1d_t gradientTex,
	int displayMethod,
	float displayScale)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	const __global real* state = stateBuffer + NUM_STATES * index;

#if DIM == 1	//pack everything into one variable
	float alpha = states[STATE_ALPHA];
	float volume = states[STATE_ALPHA] * sym33determinant(states + STATE_GAMMA);
	float4 color = (float4)(alpha, volume, K, 0.f) * displayScale;
#else			//pick a single gradient and map it to a palette
	float value = 0.f;
	switch (displayMethod) {
	case DISPLAY_METHOD_LAPSE:
		value = states[STATE_ALPHA];
		break;
	case DISPLAY_METHOD_VOLUME:
		value = states[STATE_ALPHA] * sym33determinant(states + STATE_GAMMA);
		break;
	case DISPLAY_EXTRINSIC_CURVATURE:
		{
			float gammaInverse[6];
			sym33inverse(states + STATE_GAMMA, gammaInverse);
			for (int i = 0; i < 3; ++i) {	
				//single pass of the diagonals
				value += states[STATE_K_XX + i] * gammaInverse[i + SYM33_XX];
				//double pass of the skew entries (which are symmetric) ... (not skew-symmetric (i.e. antisymmetric))
				value += 2.f * states[STATE_K_XY + i] * gammaInverse[i + SYM33_XY];
			}
		}
		break;
	}
	value *= displayScale;
	float4 color = read_imagef(gradientTex, CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, value);
#endif
	write_imagef(fluidTex, (int4)(i.x, i.y, i.z, 0), color);
}

constant float2 offset[6] = {
	(float2)(-.5f, 0.f),
	(float2)(.5f, 0.f),
	(float2)(.2f, .3f),
	(float2)(.5f, 0.f),
	(float2)(.2f, -.3f),
	(float2)(.5f, 0.f),
};

__kernel void updateVectorField(
	__global real* vectorFieldVertexBuffer,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	float scale)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int4 size = (int4)(get_global_size(0), get_global_size(1), get_global_size(2), 0);	
	int vertexIndex = i.x + size.x * (i.y + size.y * i.z);
	__global real* vertex = vectorFieldVertexBuffer + 6 * 3 * vertexIndex;
	
	float4 f = (float4)(
		((float)i.x + .5f) / (float)size.x,
		((float)i.y + .5f) / (float)size.y,
		((float)i.z + .5f) / (float)size.z,
		0.f);

	//times grid size divided by velocity field size
	float4 sf = (float4)(f.x * SIZE_X, f.y * SIZE_Y, f.z * SIZE_Z, 0.f);
	int4 si = (int4)(sf.x, sf.y, sf.z, 0);
	//float4 fp = (float4)(sf.x - (float)si.x, sf.y - (float)si.y, sf.z - (float)si.z, 0.f);
	
#if 1	//plotting velocity 
	int stateIndex = INDEXV(si);
	const __global real* state = stateBuffer + NUM_STATES * stateIndex;
	float4 velocity = (float4)(state[0], 0.f, 0.f, 0.f);	//extrinsic curvature?  what's velocity?
#endif

	//velocity is the first axis of the basis to draw the arrows
	//the second should be perpendicular to velocity
#if DIM < 3
	real4 tv = (real4)(-velocity.y, velocity.x, 0.f, 0.f);
#elif DIM == 3
	real4 vx = (real4)(0.f, -velocity.z, velocity.y, 0.f);
	real4 vy = (real4)(velocity.z, 0.f, -velocity.x, 0.f);
	real4 vz = (real4)(-velocity.y, velocity.x, 0.f, 0.f);
	real lxsq = dot(vx,vx);
	real lysq = dot(vy,vy);
	real lzsq = dot(vz,vz);
	real4 tv;
	if (lxsq > lysq) {	//x > y
		if (lxsq > lzsq) {	//x > z, x > y
			tv = vx;
		} else {	//z > x > y
			tv = vz;
		}
	} else {	//y >= x
		if (lysq > lzsq) {	//y >= x, y > z
			tv = vy;
		} else {	// z > y >= x
			tv = vz;
		}
	}
#endif

	for (int i = 0; i < 6; ++i) {
		vertex[0 + 3 * i] = f.x * (XMAX - XMIN) + XMIN + scale * (offset[i].x * velocity.x + offset[i].y * tv.x);
		vertex[1 + 3 * i] = f.y * (YMAX - YMIN) + YMIN + scale * (offset[i].x * velocity.y + offset[i].y * tv.y);
		vertex[2 + 3 * i] = f.z * (ZMAX - ZMIN) + ZMIN + scale * (offset[i].x * velocity.z + offset[i].y * tv.z);
	}
}


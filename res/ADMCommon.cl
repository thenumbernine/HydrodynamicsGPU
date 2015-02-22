#include "HydroGPU/Shared/Common.h"

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

	real alpha = state[STATE_ALPHA];
	real g = state[STATE_G];
	real K = state[STATE_K];
	
	float4 color = (float4)(alpha, g, K, 0.f) * displayScale;
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
	float4 velocity = (float4)(state[4], 0.f, 0.f, 0.f);	//extrinsic curvature?  what's velocity?
#endif
#if 0	//plotting gravity
	int4 ixL = si; ixL.x = (ixL.x + SIZE_X - 1) % SIZE_X;
	int4 ixR = si; ixR.x = (ixR.x + 1) % SIZE_X;
	int4 iyL = si; iyL.y = (iyL.y + SIZE_X - 1) % SIZE_X;
	int4 iyR = si; iyR.y = (iyR.y + 1) % SIZE_X;
	//external force is negative the potential gradient
	float4 velocity = (float4)(
		gravityPotentialBuffer[INDEXV(ixL)] - gravityPotentialBuffer[INDEXV(ixR)],
		gravityPotentialBuffer[INDEXV(iyL)] - gravityPotentialBuffer[INDEXV(iyR)],
		0.f,
		0.f);
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


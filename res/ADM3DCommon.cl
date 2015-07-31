#include "HydroGPU/Shared/Common.h"
#include "HydroGPU/ADM3D.h"

real det3x3sym(real xx, real xy, real xz, real yy, real yz, real zz) {
	return xx * yy * zz
		+ xy * yz * xz
		+ xz * xy * yz
		- xz * yy * xz
		- yz * yz * xx
		- zz * xy * xy;
}
	
real8 inv3x3sym(real xx, real xy, real xz, real yy, real yz, real zz, real det) {
	return (real8)(
		(yy * zz - yz * yz) / det,		// xx
		(xz * yz - xy * zz)  / det,	// xy
		(xy * yz - xz * yy) / det,	// xz
		(xx * zz - xz * xz) / det,		// yy
		(xz * xy - xx * yz) / det,	// yz
		(xx * yy - xy * xy) / det,		// zz
		0.f, 0.f);
}

//specific to Euler equations
__kernel void convertToTex(
	__write_only image3d_t destTex,
	int displayMethod,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	const __global real* state = stateBuffer + NUM_STATES * index;

	real alpha = state[0];
	real g_xx = state[1], g_xy = state[2], g_xz = state[3], g_yy = state[4], g_yz = state[5], g_zz = state[6];
	//real A_x = state[7], A_y = state[8], A_z = state[9];
	//real D_xxx = state[10], D_xxy = state[11], D_xxz = state[12], D_xyy = state[13], D_xyz = state[14], D_xzz = state[15];
	//real D_yxx = state[16], D_yxy = state[17], D_yxz = state[18], D_yyy = state[19], D_yyz = state[20], D_yzz = state[21];
	//real D_zxx = state[22], D_zxy = state[23], D_zxz = state[24], D_zyy = state[25], D_zyz = state[26], D_zzz = state[27];
	real K_xx = state[28], K_xy = state[29], K_xz = state[30], K_yy = state[31], K_yz = state[32], K_zz = state[33];
	real g = det3x3sym(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz);
	real8 gInv = inv3x3sym(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz, g);
	real gUxx = gInv[0], gUxy = gInv[1], gUxz = gInv[2], gUyy = gInv[3], gUyz = gInv[4], gUzz = gInv[5];
	real tr_K = K_xx * gUxx + K_yy * gUyy + K_zz * gUzz + 2.f * K_xy * gUxy + 2.f * K_yz * gUyz + 2.f * K_xz * gUxz;
	
	float value = 0.f;
	switch (displayMethod) {
	case DISPLAY_ALPHA:
		value = alpha;
		break;
	case DISPLAY_VOLUME:
		value = alpha * g; 
		break;
	case DISPLAY_K:
		value = tr_K;
		break;
	}
	
	write_imagef(destTex, (int4)(i.x, i.y, i.z, 0), (float4)(value, 0.f, 0.f, 0.f));
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


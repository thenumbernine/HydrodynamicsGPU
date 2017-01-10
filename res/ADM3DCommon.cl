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
		0., 0.);
}

//specific to Euler equations
__kernel void convertToTex(
#ifdef has_gl_sharing 
	__write_only image3d_t destTex,
#else
	global float4* destTex,
#endif
	int displayMethod,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	const __global real* state = stateBuffer + NUM_STATES * index;

	real alpha = state[0];
	real gamma_xx = state[1], gamma_xy = state[2], gamma_xz = state[3], gamma_yy = state[4], gamma_yz = state[5], gamma_zz = state[6];
	real A_x = state[7], A_y = state[8], A_z = state[9];
	real /*D_xxx = state[10], */D_xxy = state[11], D_xxz = state[12], D_xyy = state[13], D_xyz = state[14], D_xzz = state[15];
	real D_yxx = state[16], D_yxy = state[17], D_yxz = state[18]/*, D_yyy = state[19]*/, D_yyz = state[20], D_yzz = state[21];
	real D_zxx = state[22], D_zxy = state[23], D_zxz = state[24], D_zyy = state[25], D_zyz = state[26]/*, D_zzz = state[27]*/;
	real K_xx = state[28], K_xy = state[29], K_xz = state[30], K_yy = state[31], K_yz = state[32], K_zz = state[33];
	real V_x = state[34], V_y = state[35], V_z = state[36];

	float value = 0.;
	if (displayMethod == DISPLAY_VOLUME) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		value = alpha * sqrt(gamma);
	} else if (displayMethod == DISPLAY_K) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
		real tr_K = K_xx * gammaUxx + K_yy * gammaUyy + K_zz * gammaUzz + 2. * K_xy * gammaUxy + 2. * K_yz * gammaUyz + 2. * K_xz * gammaUxz;
		value = tr_K;
	//TODO incorporate beta into the gravity equation.
	//-Gamma^k_tt = (-alpha alpha,j - beta_j,t + beta^i beta_i,j) gamma^jk + (beta^k alpha,t + beta^k beta^j alpha,j) / alpha - (beta^i beta^j beta^k beta_i,j) / alpha^2
	} else if (displayMethod == DISPLAY_GRAVITY_MAGN) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
		real4 AU = (real4)(
			A_x * gammaUxx + A_y * gammaUxy + A_z * gammaUxz,
			A_x * gammaUxy + A_y * gammaUyy + A_z * gammaUyz,
			A_x * gammaUxz + A_y * gammaUyz + A_z * gammaUzz,
			0.);
		value = alpha * alpha * length(AU);
	} else if (displayMethod == DISPLAY_EXPANSION) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
		real tr_K = K_xx * gammaUxx + K_yy * gammaUyy + K_zz * gammaUzz + 2. * K_xy * gammaUxy + 2. * K_yz * gammaUyz + 2. * K_xz * gammaUxz;
		value = -alpha * tr_K;
/*	} else if (displayMethod == DISPLAY_GAUSSIAN_CURVATURE) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
		real tr_K = K_xx * gammaUxx + K_yy * gammaUyy + K_zz * gammaUzz + 2. * K_xy * gammaUxy + 2. * K_yz * gammaUyz + 2. * K_xz * gammaUxz;
		//R4 = R3 + K_ij K^ij - K^2
		// = R3 + tr(K^i_k K^k_j) - tr(K^i_j)^2
		real KUL[3][3] = {
			{gammaUxx * K_xx + gammaUxy * K_xy + gammaUxz * K_xz,
			gammaUxx * K_xy + gammaUxy * K_yy + gammaUxz * K_yz,
			gammaUxx * K_xz + gammaUxy * K_yz + gammaUxz * K_zz},
			{gammaUxy * K_xx + gammaUyy * K_xy + gammaUyz * K_xz,
			gammaUxy * K_xy + gammaUyy * K_yy + gammaUyz * K_yz,
			gammaUxy * K_xz + gammaUyy * K_yz + gammaUyz * K_zz},
			{gammaUxz * K_xx + gammaUyz * K_xy + gammaUzz * K_xz,
			gammaUxz * K_xy + gammaUyz * K_yy + gammaUzz * K_yz,
			gammaUxz * K_xz + gammaUyz * K_yz + gammaUzz * K_zz}};
		real tr_KSq = 0;
		for (int i = 0; i < 3; ++j) {
			for (int j = 0; j < 3; ++k) {
				tr_KSq += KUL[i][j] * KUL[j][i];
			}
		}
		value = R + tr_KSq - tr_K * tr_K;
*/
	} else if (displayMethod == DISPLAY_GAMMA) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		value = gamma;

	//V_k = D_km^m - D^m_mk = (D_kmn - D_mnk) gamma^mn
	} else if (displayMethod == DISPLAY_V_CONSTRAINT_X) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real /*gammaUxx = gammaInv[0], */gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
		value = ((((((V_x - (gammaUxy * D_xxy)) - (gammaUxz * D_xxz)) - (gammaUyy * D_xyy)) - (2. * gammaUyz * D_xyz)) - (gammaUzz * D_xzz)) + (gammaUxy * D_yxx) + (gammaUxz * D_zxx) + (gammaUyy * D_yxy) + (gammaUyz * D_zxy) + (gammaUyz * D_yxz) + (gammaUzz * D_zxz));
	} else if (displayMethod == DISPLAY_V_CONSTRAINT_Y) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], /*gammaUyy = gammaInv[3], */gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
		value = ((((((V_y - (gammaUxx * D_yxx)) - (gammaUxy * D_yxy)) - (2. * gammaUxz * D_yxz)) - (gammaUyz * D_yyz)) - (gammaUzz * D_yzz)) + (gammaUxx * D_xxy) + (gammaUxz * D_zxy) + (gammaUxy * D_xyy) + (gammaUyz * D_zyy) + (gammaUxz * D_xyz) + (gammaUzz * D_zyz));
	} else if (displayMethod == DISPLAY_V_CONSTRAINT_Z) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4]/*, gammaUzz = gammaInv[5]*/;
		value = ((((((V_z - (gammaUxx * D_zxx)) - (2. * gammaUxy * D_zxy)) - (gammaUxz * D_zxz)) - (gammaUyy * D_zyy)) - (gammaUyz * D_zyz)) + (gammaUxx * D_xxz) + (gammaUxy * D_yxz) + (gammaUxy * D_xyz) + (gammaUyy * D_yyz) + (gammaUxz * D_xzz) + (gammaUyz * D_yzz));

	//states
	} else if (displayMethod < NUM_STATES) {
		value = state[displayMethod];

	//derivative constraints
	} else  if (i.x < 1 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 1 || i.y >= SIZE_Y - 1
#endif
#if DIM > 2
		|| i.z < 1 || i.z >= SIZE_Z - 1
#endif
	) {

		//these constraints seem to be fulfilled pretty easily
		//the V constraint ... not so much ... and when I enforce it, I get bad values ... 

		//A_i = (ln alpha),i = alpha,i / alpha
		//alpha A_i = alpha,i
		if (displayMethod >= DISPLAY_A_ALPHA_CONSTRAINT_X && displayMethod < DISPLAY_A_ALPHA_CONSTRAINT_Z) {
			int side = displayMethod - DISPLAY_A_ALPHA_CONSTRAINT_X;
			
			int indexL = index - NUM_STATES * stepsize[side];
			int indexR = index + NUM_STATES * stepsize[side];
			const __global real* stateL = stateBuffer + NUM_STATES * indexL;
			const __global real* stateR = stateBuffer + NUM_STATES * indexR;
			
			real dalpha_dx = (stateR[STATE_ALPHA] - stateL[STATE_ALPHA]) / (2. * dx[side]);
			value = alpha * state[STATE_A_X+side] - dalpha_dx;

		//D_kij = 1/2 gamma_ij,k 
		} else if (displayMethod >= DISPLAY_D_X_GAMMA_CONSTRAINT_XX && displayMethod < DISPLAY_D_Z_GAMMA_CONSTRAINT_ZZ) {
			int index18 = displayMethod - DISPLAY_D_X_GAMMA_CONSTRAINT_XX;
			int side = index18 / 6;
			int ij = index18 - 6 * side;

			int indexL = index - NUM_STATES * stepsize[side];
			int indexR = index + NUM_STATES * stepsize[side];
			const __global real* stateL = stateBuffer + NUM_STATES * indexL;
			const __global real* stateR = stateBuffer + NUM_STATES * indexR;

			real dgamma_dx = (stateR[STATE_GAMMA_XX+ij] - stateL[STATE_GAMMA_XX+ij]) / (2. * dx[side]);
			value = state[STATE_D_XXX+index18] - .5 * dgamma_dx;
		}
	
	} //TODO else show some kind of garbage pattern

#ifdef has_gl_sharing 
	write_imagef(destTex, (int4)(i.x, i.y, i.z, 0), (float4)(value, 0., 0., 0.));
#else
	destTex[index] = (float4)(value, 0., 0., 0.);
#endif
}

constant float2 offset[6] = {
	(float2)(-.5, 0.),
	(float2)(.5, 0.),
	(float2)(.2, .3),
	(float2)(.5, 0.),
	(float2)(.2, -.3),
	(float2)(.5, 0.),
};

__kernel void updateVectorField(
	__global float* vectorFieldVertexBuffer,
	real scale,
	int displayMethod,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int4 size = (int4)(get_global_size(0), get_global_size(1), get_global_size(2), 0);	
	int vertexIndex = i.x + size.x * (i.y + size.y * i.z);
	__global float* vertex = vectorFieldVertexBuffer + 6 * 3 * vertexIndex;
		
	float4 f = (float4)(
		((float)i.x + .5) / (float)size.x,
		((float)i.y + .5) / (float)size.y,
		((float)i.z + .5) / (float)size.z,
		0.);

	//times grid size divided by velocity field size
	float4 sf = (float4)(f.x * SIZE_X, f.y * SIZE_Y, f.z * SIZE_Z, 0.);
	int4 si = (int4)(sf.x, sf.y, sf.z, 0);
	//float4 fp = (float4)(sf.x - (float)si.x, sf.y - (float)si.y, sf.z - (float)si.z, 0.);
	
	int stateIndex = INDEXV(si);
	const __global real* state = stateBuffer + NUM_STATES * stateIndex;
	
	real alpha = state[0];
	real gamma_xx = state[1], gamma_xy = state[2], gamma_xz = state[3], gamma_yy = state[4], gamma_yz = state[5], gamma_zz = state[6];
	real A_x = state[7], A_y = state[8], A_z = state[9];
	real /*D_xxx = state[10], */D_xxy = state[11], D_xxz = state[12], D_xyy = state[13], D_xyz = state[14], D_xzz = state[15];
	real D_yxx = state[16], D_yxy = state[17], D_yxz = state[18]/*, D_yyy = state[19]*/, D_yyz = state[20], D_yzz = state[21];
	real D_zxx = state[22], D_zxy = state[23], D_zxz = state[24], D_zyy = state[25], D_zyz = state[26]/*, D_zzz = state[27]*/;
	real K_xx = state[28], K_xy = state[29], K_xz = state[30], K_yy = state[31], K_yz = state[32], K_zz = state[33];
	real V_x = state[34], V_y = state[35], V_z = state[36];

	real4 field;
	if (displayMethod == VECTORFIELD_GAMMA_X) {
		field = (real4)(gamma_xx, gamma_xy, gamma_xz, 0.);
	} else if (displayMethod == VECTORFIELD_GAMMA_Y) {
		field = (real4)(gamma_xy, gamma_yy, gamma_yz, 0.);
	} else if (displayMethod == VECTORFIELD_GAMMA_Z) {
		field = (real4)(gamma_xz, gamma_yz, gamma_zz, 0.);
	} else if (displayMethod == VECTORFIELD_A) {
		field = (real4)(A_x, A_y, A_z, 0.);
	} else if (displayMethod == VECTORFIELD_K_X) {
		field = (real4)(K_xx, K_xy, K_xz, 0.);
	} else if (displayMethod == VECTORFIELD_K_Y) {
		field = (real4)(K_xy, K_yy, K_yz, 0.);
	} else if (displayMethod == VECTORFIELD_K_Z) {
		field = (real4)(K_xz, K_yz, K_zz, 0.);
	} else if (displayMethod == VECTORFIELD_V) {
		field = (real4)(V_x, V_y, V_z, 0.);
	} else if (displayMethod == VECTORFIELD_GRAVITY) {
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];

		real4 AU = (real4)(
			A_x * gammaUxx + A_y * gammaUxy + A_z * gammaUxz,
			A_x * gammaUxy + A_y * gammaUyy + A_z * gammaUyz,
			A_x * gammaUxz + A_y * gammaUyz + A_z * gammaUzz,
			0.);
		//TODO incorporate betas ...
		field = -alpha * alpha * AU;
	//} else if (displayMethod == VECTORFIELD_TIDAL) {
		//TODO
	} else if (displayMethod == VECTORFIELD_V_CONSTRAINT) {
		//V_k = D_km^m - D^m_mk = (D_kmn - D_mnk) gamma^mn
		real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
		real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
		real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
		field.x = ((((((V_x - (gammaUxy * D_xxy)) - (gammaUxz * D_xxz)) - (gammaUyy * D_xyy)) - (2. * gammaUyz * D_xyz)) - (gammaUzz * D_xzz)) + (gammaUxy * D_yxx) + (gammaUxz * D_zxx) + (gammaUyy * D_yxy) + (gammaUyz * D_zxy) + (gammaUyz * D_yxz) + (gammaUzz * D_zxz));
		field.y = ((((((V_y - (gammaUxx * D_yxx)) - (gammaUxy * D_yxy)) - (2. * gammaUxz * D_yxz)) - (gammaUyz * D_yyz)) - (gammaUzz * D_yzz)) + (gammaUxx * D_xxy) + (gammaUxz * D_zxy) + (gammaUxy * D_xyy) + (gammaUyz * D_zyy) + (gammaUxz * D_xyz) + (gammaUzz * D_zyz));
		field.z = ((((((V_z - (gammaUxx * D_zxx)) - (2. * gammaUxy * D_zxy)) - (gammaUxz * D_zxz)) - (gammaUyy * D_zyy)) - (gammaUyz * D_zyz)) + (gammaUxx * D_xxz) + (gammaUxy * D_yxz) + (gammaUxy * D_xyz) + (gammaUyy * D_yyz) + (gammaUxz * D_xzz) + (gammaUyz * D_yzz));
	//} else if (displayMethod == VECTORFIELD_MOMENTUM_CONSTRAINT) {
		//TODO
	}

	//field is the first axis of the basis to draw the arrows
	//the second should be perpendicular to field
#if DIM < 3
	real4 tv = (real4)(-field.y, field.x, 0., 0.);
#elif DIM == 3
	real4 vx = (real4)(0., -field.z, field.y, 0.);
	real4 vy = (real4)(field.z, 0., -field.x, 0.);
	real4 vz = (real4)(-field.y, field.x, 0., 0.);
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
		vertex[0 + 3 * i] = f.x * (XMAX - XMIN) + XMIN + scale * (offset[i].x * field.x + offset[i].y * tv.x);
		vertex[1 + 3 * i] = f.y * (YMAX - YMIN) + YMIN + scale * (offset[i].x * field.y + offset[i].y * tv.y);
		vertex[2 + 3 * i] = f.z * (ZMAX - ZMIN) + ZMIN + scale * (offset[i].x * field.z + offset[i].y * tv.z);
	}
}


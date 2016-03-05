/*
The components of the Roe solver specific to the ADM equations
paritcularly the spectral decomposition

This is the 3D version

looks like it will require refining from the Euler Roe
in which the Euler Roe eigenbasis operates on all state variables
whereas the ADM Roe only operates on certain ones ...

check out symmath/tests/numerical_relativity_codegen.lua for my derivation
*/

#include "HydroGPU/Shared/Common.h"
#include "HydroGPU/ADM3D.h"

__kernel void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenfieldsBuffer,
	const __global real* stateBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) return;
	int index = INDEXV(i);

	int indexPrev = index - stepsize[side];

	int interfaceIndex = index;
	
	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* eigenfield = eigenfieldsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;

	//store the intermediate state in the eigenfield and reconstruct it upon eigenfield[Inverse]Transform()
	for (int i = 0; i < NUM_STATES; ++i) {
		eigenfield[i] = .5f * (stateL[i] + stateR[i]);
	}

	real alpha = eigenfield[0];
	real gamma_xx = eigenfield[1], gamma_xy = eigenfield[2], gamma_xz = eigenfield[3], gamma_yy = eigenfield[4], gamma_yz = eigenfield[5], gamma_zz = eigenfield[6];
	//real A_x = eigenfield[7], A_y = eigenfield[8], A_z = eigenfield[9];
	//real D_xxx = eigenfield[10], D_xxy = eigenfield[11], D_xxz = eigenfield[12], D_xyy = eigenfield[13], D_xyz = eigenfield[14], D_xzz = eigenfield[15];
	//real D_yxx = eigenfield[16], D_yxy = eigenfield[17], D_yxz = eigenfield[18], D_yyy = eigenfield[19], D_yyz = eigenfield[20], D_yzz = eigenfield[21];
	//real D_zxx = eigenfield[22], D_zxy = eigenfield[23], D_zxz = eigenfield[24], D_zyy = eigenfield[25], D_zyz = eigenfield[26], D_zzz = eigenfield[27];
	//real K_xx = eigenfield[28], K_xy = eigenfield[29], K_xz = eigenfield[30], K_yy = eigenfield[31], K_yz = eigenfield[32], K_zz = eigenfield[33];
	//real V_x = eigenfield[34], V_y = eigenfield[35], V_z = eigenfield[36];
	real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
	real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
	real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
	real f = ADM_BONA_MASSO_F;	//could be based on alpha...

	//store only what information is needed for the function of applying the eigenvectors/inverses 
	eigenfield[37] = gammaUxx;
	eigenfield[38] = gammaUxy;
	eigenfield[39] = gammaUxz;
	eigenfield[40] = gammaUyy;
	eigenfield[41] = gammaUyz;
	eigenfield[42] = gammaUzz;
	eigenfield[43] = gamma;
	eigenfield[44] = f;

	//eigenvalues

	real lambdaLight;
	switch (side) {
	case 0:
		lambdaLight = alpha * sqrt(gammaUxx); 
		break;
	case 1:
		lambdaLight = alpha * sqrt(gammaUyy); 
		break;
	case 2:
		lambdaLight = alpha * sqrt(gammaUzz); 
		break;
	}
	real lambdaGauge = lambdaLight * sqrt(f);
	eigenvalues[0] = -lambdaGauge;
	eigenvalues[1] = -lambdaLight;
	eigenvalues[2] = -lambdaLight;
	eigenvalues[3] = -lambdaLight;
	eigenvalues[4] = -lambdaLight;
	eigenvalues[5] = -lambdaLight;
	eigenvalues[6] = 0.f;
	eigenvalues[7] = 0.f;
	eigenvalues[8] = 0.f;
	eigenvalues[9] = 0.f;
	eigenvalues[10] = 0.f;
	eigenvalues[11] = 0.f;
	eigenvalues[12] = 0.f;
	eigenvalues[13] = 0.f;
	eigenvalues[14] = 0.f;
	eigenvalues[15] = 0.f;
	eigenvalues[16] = 0.f;
	eigenvalues[17] = 0.f;
	eigenvalues[18] = 0.f;
	eigenvalues[19] = 0.f;
	eigenvalues[20] = 0.f;
	eigenvalues[21] = 0.f;
	eigenvalues[22] = 0.f;
	eigenvalues[23] = 0.f;
	eigenvalues[24] = 0.f;
	eigenvalues[25] = 0.f;
	eigenvalues[26] = 0.f;
	eigenvalues[27] = 0.f;
	eigenvalues[28] = 0.f;
	eigenvalues[29] = 0.f;
	eigenvalues[30] = 0.f;
	eigenvalues[31] = lambdaLight;
	eigenvalues[32] = lambdaLight;
	eigenvalues[33] = lambdaLight;
	eigenvalues[34] = lambdaLight;
	eigenvalues[35] = lambdaLight;
	eigenvalues[36] = lambdaGauge;
}

void eigenfieldTransform(
	real* results,
	const __global real* eigenfield,
	const real* input,
	int side)
{
	//real alpha = eigenfield[0];
	//real gamma_xx = eigenfield[1], gamma_xy = eigenfield[2], gamma_xz = eigenfield[3], gamma_yy = eigenfield[4], gamma_yz = eigenfield[5], gamma_zz = eigenfield[6];
	//real A_x = eigenfield[7], A_y = eigenfield[8], A_z = eigenfield[9];
	//real D_xxx = eigenfield[10], D_xxy = eigenfield[11], D_xxz = eigenfield[12], D_xyy = eigenfield[13], D_xyz = eigenfield[14], D_xzz = eigenfield[15];
	//real D_yxx = eigenfield[16], D_yxy = eigenfield[17], D_yxz = eigenfield[18], D_yyy = eigenfield[19], D_yyz = eigenfield[20], D_yzz = eigenfield[21];
	//real D_zxx = eigenfield[22], D_zxy = eigenfield[23], D_zxz = eigenfield[24], D_zyy = eigenfield[25], D_zyz = eigenfield[26], D_zzz = eigenfield[27];
	//real K_xx = eigenfield[28], K_xy = eigenfield[29], K_xz = eigenfield[30], K_yy = eigenfield[31], K_yz = eigenfield[32], K_zz = eigenfield[33];
	//real V_x = eigenfield[34], V_y = eigenfield[35], V_z = eigenfield[36];
	real gammaUxx = eigenfield[37], gammaUxy = eigenfield[38], gammaUxz = eigenfield[39], gammaUyy = eigenfield[40], gammaUyz = eigenfield[41], gammaUzz = eigenfield[42];
	//real gamma = eigenfield[43];
	real f = eigenfield[44];

	real sqrt_f = sqrt(f);
	if (side == 0) {
		real sqrt_gammaUxx = sqrt(gammaUxx);
		real gammaUxx_toThe_3_2 = sqrt_gammaUxx * gammaUxx;

		results[0] = ((((-(2.f * gammaUxz * input[36])) - (gammaUxx * input[7])) + (sqrt_f * gammaUxx_toThe_3_2 * input[28]) + (2.f * sqrt_f * gammaUxy * input[29] * sqrt_gammaUxx) + (2.f * sqrt_f * gammaUxz * input[30] * sqrt_gammaUxx) + (sqrt_f * gammaUyy * input[31] * sqrt_gammaUxx) + (2.f * sqrt_f * gammaUyz * input[32] * sqrt_gammaUxx) + (((sqrt_f * gammaUzz * input[33] * sqrt_gammaUxx) - (2.f * gammaUxx * input[34])) - (2.f * gammaUxy * input[35]))) / sqrt_gammaUxx);
		results[1] = (((-(gammaUxx * input[11])) + ((input[29] * sqrt_gammaUxx) - input[35])) / sqrt_gammaUxx);
		results[2] = (((-(gammaUxx * input[12])) + ((input[30] * sqrt_gammaUxx) - input[36])) / sqrt_gammaUxx);
		results[3] = ((-(sqrt_gammaUxx * input[13])) + input[31]);
		results[4] = ((-(sqrt_gammaUxx * input[14])) + input[32]);
		results[5] = ((-(sqrt_gammaUxx * input[15])) + input[33]);
		results[6] = input[0];
		results[7] = input[1];
		results[8] = input[2];
		results[9] = input[3];
		results[10] = input[4];
		results[11] = input[5];
		results[12] = input[6];
		results[13] = input[8];
		results[14] = input[9];
		results[15] = input[16];
		results[16] = input[17];
		results[17] = input[18];
		results[18] = input[19];
		results[19] = input[20];
		results[20] = input[21];
		results[21] = input[22];
		results[22] = input[23];
		results[23] = input[24];
		results[24] = input[25];
		results[25] = input[26];
		results[26] = input[27];
		results[27] = input[34];
		results[28] = input[35];
		results[29] = input[36];
		results[30] = ((((((input[7] - (f * gammaUxx * input[10])) - (2.f * f * gammaUxy * input[11])) - (2.f * f * gammaUxz * input[12])) - (f * gammaUyy * input[13])) - (2.f * f * gammaUyz * input[14])) - (f * gammaUzz * input[15]));
		results[31] = (((gammaUxx * input[11]) + (input[29] * sqrt_gammaUxx) + input[35]) / sqrt_gammaUxx);
		results[32] = (((gammaUxx * input[12]) + (input[30] * sqrt_gammaUxx) + input[36]) / sqrt_gammaUxx);
		results[33] = ((sqrt_gammaUxx * input[13]) + input[31]);
		results[34] = ((sqrt_gammaUxx * input[14]) + input[32]);
		results[35] = ((sqrt_gammaUxx * input[15]) + input[33]);
		results[36] = (((2.f * gammaUxz * input[36]) + (gammaUxx * input[7]) + (sqrt_f * gammaUxx_toThe_3_2 * input[28]) + (2.f * sqrt_f * gammaUxy * input[29] * sqrt_gammaUxx) + (2.f * sqrt_f * gammaUxz * input[30] * sqrt_gammaUxx) + (sqrt_f * gammaUyy * input[31] * sqrt_gammaUxx) + (2.f * sqrt_f * gammaUyz * input[32] * sqrt_gammaUxx) + (sqrt_f * gammaUzz * input[33] * sqrt_gammaUxx) + (2.f * gammaUxx * input[34]) + (2.f * gammaUxy * input[35])) / sqrt_gammaUxx);
	} else if (side == 1) {
		real sqrt_gammaUyy = sqrt(gammaUyy);
		real gammaUyy_toThe_3_2 = sqrt_gammaUyy * gammaUyy;

		results[0] = (((((-(2.f * gammaUyz * input[36])) - (2.f * gammaUyy * input[35])) - (gammaUyy * input[8])) + (sqrt_f * gammaUxx * input[28] * sqrt_gammaUyy) + (2.f * sqrt_f * gammaUxy * input[29] * sqrt_gammaUyy) + (2.f * sqrt_f * gammaUxz * input[30] * sqrt_gammaUyy) + (sqrt_f * gammaUyy_toThe_3_2 * input[31]) + (2.f * sqrt_f * gammaUyz * input[32] * sqrt_gammaUyy) + ((sqrt_f * gammaUzz * input[33] * sqrt_gammaUyy) - (2.f * gammaUxy * input[34]))) / sqrt_gammaUyy);
		results[1] = ((-(sqrt_gammaUyy * input[16])) + input[28]);
		results[2] = (((-(gammaUyy * input[17])) + ((input[29] * sqrt_gammaUyy) - input[34])) / sqrt_gammaUyy);
		results[3] = ((-(sqrt_gammaUyy * input[18])) + input[30]);
		results[4] = (((-(gammaUyy * input[20])) + ((input[32] * sqrt_gammaUyy) - input[36])) / sqrt_gammaUyy);
		results[5] = ((-(sqrt_gammaUyy * input[21])) + input[33]);
		results[6] = input[0];
		results[7] = input[1];
		results[8] = input[2];
		results[9] = input[3];
		results[10] = input[4];
		results[11] = input[5];
		results[12] = input[6];
		results[13] = input[7];
		results[14] = input[9];
		results[15] = input[10];
		results[16] = input[11];
		results[17] = input[12];
		results[18] = input[13];
		results[19] = input[14];
		results[20] = input[15];
		results[21] = input[22];
		results[22] = input[23];
		results[23] = input[24];
		results[24] = input[25];
		results[25] = input[26];
		results[26] = input[27];
		results[27] = input[34];
		results[28] = input[35];
		results[29] = input[36];
		results[30] = ((((((input[8] - (f * gammaUxx * input[16])) - (2.f * f * gammaUxy * input[17])) - (2.f * f * gammaUxz * input[18])) - (f * gammaUyy * input[19])) - (2.f * f * gammaUyz * input[20])) - (f * gammaUzz * input[21]));
		results[31] = ((sqrt_gammaUyy * input[16]) + input[28]);
		results[32] = (((gammaUyy * input[17]) + (input[29] * sqrt_gammaUyy) + input[34]) / sqrt_gammaUyy);
		results[33] = ((sqrt_gammaUyy * input[18]) + input[30]);
		results[34] = (((gammaUyy * input[20]) + (input[32] * sqrt_gammaUyy) + input[36]) / sqrt_gammaUyy);
		results[35] = ((sqrt_gammaUyy * input[21]) + input[33]);
		results[36] = (((2.f * gammaUyz * input[36]) + (2.f * gammaUyy * input[35]) + (gammaUyy * input[8]) + (sqrt_f * gammaUxx * input[28] * sqrt_gammaUyy) + (2.f * sqrt_f * gammaUxy * input[29] * sqrt_gammaUyy) + (2.f * sqrt_f * gammaUxz * input[30] * sqrt_gammaUyy) + (sqrt_f * gammaUyy_toThe_3_2 * input[31]) + (2.f * sqrt_f * gammaUyz * input[32] * sqrt_gammaUyy) + (sqrt_f * gammaUzz * input[33] * sqrt_gammaUyy) + (2.f * gammaUxy * input[34])) / sqrt_gammaUyy);
	} else if (side == 2) {
		real sqrt_gammaUzz = sqrt(gammaUzz);
		real gammaUzz_toThe_3_2 = sqrt_gammaUzz * gammaUzz;
	
		results[0] = (((((-(2.f * gammaUzz * input[36])) - (2.f * gammaUyz * input[35])) - (gammaUzz * input[9])) + (sqrt_f * gammaUxx * input[28] * sqrt_gammaUzz) + (2.f * sqrt_f * gammaUxy * input[29] * sqrt_gammaUzz) + (2.f * sqrt_f * gammaUxz * input[30] * sqrt_gammaUzz) + (sqrt_f * gammaUyy * input[31] * sqrt_gammaUzz) + (2.f * sqrt_f * gammaUyz * input[32] * sqrt_gammaUzz) + ((sqrt_f * gammaUzz_toThe_3_2 * input[33]) - (2.f * gammaUxz * input[34]))) / sqrt_gammaUzz);
		results[1] = ((-(sqrt_gammaUzz * input[22])) + input[28]);
		results[2] = ((-(sqrt_gammaUzz * input[23])) + input[29]);
		results[3] = (((-(gammaUzz * input[24])) + ((input[30] * sqrt_gammaUzz) - input[34])) / sqrt_gammaUzz);
		results[4] = ((-(sqrt_gammaUzz * input[25])) + input[31]);
		results[5] = (((-(gammaUzz * input[26])) + ((input[32] * sqrt_gammaUzz) - input[35])) / sqrt_gammaUzz);
		results[6] = input[0];
		results[7] = input[1];
		results[8] = input[2];
		results[9] = input[3];
		results[10] = input[4];
		results[11] = input[5];
		results[12] = input[6];
		results[13] = input[7];
		results[14] = input[8];
		results[15] = input[10];
		results[16] = input[11];
		results[17] = input[12];
		results[18] = input[13];
		results[19] = input[14];
		results[20] = input[15];
		results[21] = input[16];
		results[22] = input[17];
		results[23] = input[18];
		results[24] = input[19];
		results[25] = input[20];
		results[26] = input[21];
		results[27] = input[34];
		results[28] = input[35];
		results[29] = input[36];
		results[30] = ((((((input[9] - (f * gammaUxx * input[22])) - (2.f * f * gammaUxy * input[23])) - (2.f * f * gammaUxz * input[24])) - (f * gammaUyy * input[25])) - (2.f * f * gammaUyz * input[26])) - (f * gammaUzz * input[27]));
		results[31] = ((sqrt_gammaUzz * input[22]) + input[28]);
		results[32] = ((sqrt_gammaUzz * input[23]) + input[29]);
		results[33] = (((gammaUzz * input[24]) + (input[30] * sqrt_gammaUzz) + input[34]) / sqrt_gammaUzz);
		results[34] = ((sqrt_gammaUzz * input[25]) + input[31]);
		results[35] = (((gammaUzz * input[26]) + (input[32] * sqrt_gammaUzz) + input[35]) / sqrt_gammaUzz);
		results[36] = (((2.f * gammaUzz * input[36]) + (2.f * gammaUyz * input[35]) + (gammaUzz * input[9]) + (sqrt_f * gammaUxx * input[28] * sqrt_gammaUzz) + (2.f * sqrt_f * gammaUxy * input[29] * sqrt_gammaUzz) + (2.f * sqrt_f * gammaUxz * input[30] * sqrt_gammaUzz) + (sqrt_f * gammaUyy * input[31] * sqrt_gammaUzz) + (2.f * sqrt_f * gammaUyz * input[32] * sqrt_gammaUzz) + (sqrt_f * gammaUzz_toThe_3_2 * input[33]) + (2.f * gammaUxz * input[34])) / sqrt_gammaUzz);
	}
}

void eigenfieldInverseTransform(
	__global real* results,
	const __global real* eigenfield,
	const real* input,
	int side)
{
	//real alpha = eigenfield[0];
	//real gamma_xx = eigenfield[1], gamma_xy = eigenfield[2], gamma_xz = eigenfield[3], gamma_yy = eigenfield[4], gamma_yz = eigenfield[5], gamma_zz = eigenfield[6];
	//real A_x = eigenfield[7], A_y = eigenfield[8], A_z = eigenfield[9];
	//real D_xxx = eigenfield[10], D_xxy = eigenfield[11], D_xxz = eigenfield[12], D_xyy = eigenfield[13], D_xyz = eigenfield[14], D_xzz = eigenfield[15];
	//real D_yxx = eigenfield[16], D_yxy = eigenfield[17], D_yxz = eigenfield[18], D_yyy = eigenfield[19], D_yyz = eigenfield[20], D_yzz = eigenfield[21];
	//real D_zxx = eigenfield[22], D_zxy = eigenfield[23], D_zxz = eigenfield[24], D_zyy = eigenfield[25], D_zyz = eigenfield[26], D_zzz = eigenfield[27];
	//real K_xx = eigenfield[28], K_xy = eigenfield[29], K_xz = eigenfield[30], K_yy = eigenfield[31], K_yz = eigenfield[32], K_zz = eigenfield[33];
	//real V_x = eigenfield[34], V_y = eigenfield[35], V_z = eigenfield[36];
	real gammaUxx = eigenfield[37], gammaUxy = eigenfield[38], gammaUxz = eigenfield[39], gammaUyy = eigenfield[40], gammaUyz = eigenfield[41], gammaUzz = eigenfield[42];
	//real gamma = eigenfield[43];
	real f = eigenfield[44];

	real sqrt_f = sqrt(f);

	if (side == 0) {
		real sqrt_gammaUxx = sqrt(gammaUxx);
		real gammaUxx_toThe_3_2 = sqrt_gammaUxx * gammaUxx;
	
		results[0] = input[6];
		results[1] = input[7];
		results[2] = input[8];
		results[3] = input[9];
		results[4] = input[10];
		results[5] = input[11];
		results[6] = input[12];
		results[7] = (((-input[36]) + (4.f * gammaUxz * input[29] * (1.f / sqrt_gammaUxx)) + (4.f * gammaUxy * input[28] * (1.f / sqrt_gammaUxx)) + (4.f * input[27] * sqrt_gammaUxx) + input[0]) / (-(2.f * sqrt_gammaUxx)));
		results[8] = input[13];
		results[9] = input[14];
		results[10] = (((-input[36]) + (gammaUzz * input[35] * f) + (2.f * gammaUyz * input[34] * f) + (gammaUyy * input[33] * f) + (2.f * gammaUxz * input[32] * f) + (2.f * gammaUxy * input[31] * f) + (2.f * input[30] * sqrt_gammaUxx) + ((4.f * gammaUxz * input[29] * (1.f / sqrt_gammaUxx)) - (4.f * gammaUxz * f * input[29] * (1.f / sqrt_gammaUxx))) + ((4.f * gammaUxy * input[28] * (1.f / sqrt_gammaUxx)) - (4.f * gammaUxy * f * input[28] * (1.f / sqrt_gammaUxx))) + ((((((4.f * input[27] * sqrt_gammaUxx) - (gammaUzz * input[5] * f)) - (2.f * gammaUyz * input[4] * f)) - (gammaUyy * input[3] * f)) - (2.f * gammaUxz * input[2] * f)) - (2.f * gammaUxy * input[1] * f)) + input[0]) / (-(2.f * gammaUxx_toThe_3_2 * f)));
		results[11] = (((-input[31]) + (2.f * input[28] * (1.f / sqrt_gammaUxx)) + input[1]) / (-(2.f * sqrt_gammaUxx)));
		results[12] = (((-input[32]) + (2.f * input[29] * (1.f / sqrt_gammaUxx)) + input[2]) / (-(2.f * sqrt_gammaUxx)));
		results[13] = (((-input[33]) + input[3]) / (-(2.f * sqrt_gammaUxx)));
		results[14] = (((-input[34]) + input[4]) / (-(2.f * sqrt_gammaUxx)));
		results[15] = (((-input[35]) + input[5]) / (-(2.f * sqrt_gammaUxx)));
		results[16] = input[15];
		results[17] = input[16];
		results[18] = input[17];
		results[19] = input[18];
		results[20] = input[19];
		results[21] = input[20];
		results[22] = input[21];
		results[23] = input[22];
		results[24] = input[23];
		results[25] = input[24];
		results[26] = input[25];
		results[27] = input[26];
		results[28] = ((((((((((((input[36] - (gammaUzz * input[35] * sqrt_f)) - (2.f * gammaUyz * input[34] * sqrt_f)) - (gammaUyy * input[33] * sqrt_f)) - (2.f * gammaUxz * input[32] * sqrt_f)) - (2.f * gammaUxy * input[31] * sqrt_f)) - (gammaUzz * input[5] * sqrt_f)) - (2.f * gammaUyz * input[4] * sqrt_f)) - (gammaUyy * input[3] * sqrt_f)) - (2.f * gammaUxz * input[2] * sqrt_f)) - (2.f * gammaUxy * input[1] * sqrt_f)) + input[0]) / (2.f * sqrt_f * gammaUxx));
		results[29] = ((input[31] + input[1]) / 2.f);
		results[30] = ((input[32] + input[2]) / 2.f);
		results[31] = ((input[33] + input[3]) / 2.f);
		results[32] = ((input[34] + input[4]) / 2.f);
		results[33] = ((input[35] + input[5]) / 2.f);
		results[34] = input[27];
		results[35] = input[28];
		results[36] = input[29];
	} else if (side == 1) {
		real sqrt_gammaUyy = sqrt(gammaUyy);
		real gammaUyy_toThe_3_2 = sqrt_gammaUyy * gammaUyy;

		results[0] = input[6];
		results[1] = input[7];
		results[2] = input[8];
		results[3] = input[9];
		results[4] = input[10];
		results[5] = input[11];
		results[6] = input[12];
		results[7] = input[13];
		results[8] = (((-input[36]) + (4.f * gammaUyz * input[29] * (1.f / sqrt_gammaUyy)) + (4.f * input[28] * sqrt_gammaUyy) + (4.f * gammaUxy * input[27] * (1.f / sqrt_gammaUyy)) + input[0]) / (-(2.f * sqrt_gammaUyy)));
		results[9] = input[14];
		results[10] = input[15];
		results[11] = input[16];
		results[12] = input[17];
		results[13] = input[18];
		results[14] = input[19];
		results[15] = input[20];
		results[16] = (((-input[31]) + input[1]) / (-(2.f * sqrt_gammaUyy)));
		results[17] = (((-input[32]) + (2.f * input[27] * (1.f / sqrt_gammaUyy)) + input[2]) / (-(2.f * sqrt_gammaUyy)));
		results[18] = (((-input[33]) + input[3]) / (-(2.f * sqrt_gammaUyy)));
		results[19] = (((-input[36]) + (gammaUzz * input[35] * f) + (2.f * gammaUyz * input[34] * f) + (2.f * gammaUxz * input[33] * f) + (2.f * gammaUxy * input[32] * f) + (gammaUxx * input[31] * f) + (2.f * input[30] * sqrt_gammaUyy) + ((4.f * gammaUyz * input[29] * (1.f / sqrt_gammaUyy)) - (4.f * gammaUyz * f * input[29] * (1.f / sqrt_gammaUyy))) + (4.f * input[28] * sqrt_gammaUyy) + (((((((4.f * gammaUxy * input[27] * (1.f / sqrt_gammaUyy)) - (4.f * gammaUxy * f * input[27] * (1.f / sqrt_gammaUyy))) - (gammaUzz * input[5] * f)) - (2.f * gammaUyz * input[4] * f)) - (2.f * gammaUxz * input[3] * f)) - (2.f * gammaUxy * input[2] * f)) - (gammaUxx * input[1] * f)) + input[0]) / (-(2.f * gammaUyy_toThe_3_2 * f)));
		results[20] = (((-input[34]) + (2.f * input[29] * (1.f / sqrt_gammaUyy)) + input[4]) / (-(2.f * sqrt_gammaUyy)));
		results[21] = (((-input[35]) + input[5]) / (-(2.f * sqrt_gammaUyy)));
		results[22] = input[21];
		results[23] = input[22];
		results[24] = input[23];
		results[25] = input[24];
		results[26] = input[25];
		results[27] = input[26];
		results[28] = ((input[31] + input[1]) / 2.f);
		results[29] = ((input[32] + input[2]) / 2.f);
		results[30] = ((input[33] + input[3]) / 2.f);
		results[31] = ((((((((((((input[36] - (gammaUzz * input[35] * sqrt_f)) - (2.f * gammaUyz * input[34] * sqrt_f)) - (2.f * gammaUxz * input[33] * sqrt_f)) - (2.f * gammaUxy * input[32] * sqrt_f)) - (gammaUxx * input[31] * sqrt_f)) - (gammaUzz * input[5] * sqrt_f)) - (2.f * gammaUyz * input[4] * sqrt_f)) - (2.f * gammaUxz * input[3] * sqrt_f)) - (2.f * gammaUxy * input[2] * sqrt_f)) - (gammaUxx * input[1] * sqrt_f)) + input[0]) / (2.f * sqrt_f * gammaUyy));
		results[32] = ((input[34] + input[4]) / 2.f);
		results[33] = ((input[35] + input[5]) / 2.f);
		results[34] = input[27];
		results[35] = input[28];
		results[36] = input[29];
	} else if (side == 2) {
		real sqrt_gammaUzz = sqrt(gammaUzz);
		real gammaUzz_toThe_3_2 = sqrt_gammaUzz * gammaUzz;
	
		results[0] = input[6];
		results[1] = input[7];
		results[2] = input[8];
		results[3] = input[9];
		results[4] = input[10];
		results[5] = input[11];
		results[6] = input[12];
		results[7] = input[13];
		results[8] = input[14];
		results[9] = (((-input[36]) + (4.f * input[29] * sqrt_gammaUzz) + (4.f * gammaUyz * input[28] * (1.f / sqrt_gammaUzz)) + (4.f * gammaUxz * input[27] * (1.f / sqrt_gammaUzz)) + input[0]) / (-(2.f * sqrt_gammaUzz)));
		results[10] = input[15];
		results[11] = input[16];
		results[12] = input[17];
		results[13] = input[18];
		results[14] = input[19];
		results[15] = input[20];
		results[16] = input[21];
		results[17] = input[22];
		results[18] = input[23];
		results[19] = input[24];
		results[20] = input[25];
		results[21] = input[26];
		results[22] = (((-input[31]) + input[1]) / (-(2.f * sqrt_gammaUzz)));
		results[23] = (((-input[32]) + input[2]) / (-(2.f * sqrt_gammaUzz)));
		results[24] = (((-input[33]) + (2.f * input[27] * (1.f / sqrt_gammaUzz)) + input[3]) / (-(2.f * sqrt_gammaUzz)));
		results[25] = (((-input[34]) + input[4]) / (-(2.f * sqrt_gammaUzz)));
		results[26] = (((-input[35]) + (2.f * input[28] * (1.f / sqrt_gammaUzz)) + input[5]) / (-(2.f * sqrt_gammaUzz)));
		results[27] = (((-input[36]) + (2.f * gammaUyz * input[35] * f) + (gammaUyy * input[34] * f) + (2.f * gammaUxz * input[33] * f) + (2.f * gammaUxy * input[32] * f) + (gammaUxx * input[31] * f) + (2.f * input[30] * sqrt_gammaUzz) + (4.f * input[29] * sqrt_gammaUzz) + ((4.f * gammaUyz * input[28] * (1.f / sqrt_gammaUzz)) - (4.f * gammaUyz * f * input[28] * (1.f / sqrt_gammaUzz))) + (((((((4.f * gammaUxz * input[27] * (1.f / sqrt_gammaUzz)) - (4.f * gammaUxz * f * input[27] * (1.f / sqrt_gammaUzz))) - (2.f * gammaUyz * input[5] * f)) - (gammaUyy * input[4] * f)) - (2.f * gammaUxz * input[3] * f)) - (2.f * gammaUxy * input[2] * f)) - (gammaUxx * input[1] * f)) + input[0]) / (-(2.f * gammaUzz_toThe_3_2 * f)));
		results[28] = ((input[31] + input[1]) / 2.f);
		results[29] = ((input[32] + input[2]) / 2.f);
		results[30] = ((input[33] + input[3]) / 2.f);
		results[31] = ((input[34] + input[4]) / 2.f);
		results[32] = ((input[35] + input[5]) / 2.f);
		results[33] = ((((((((((((input[36] - (2.f * gammaUyz * input[35] * sqrt_f)) - (gammaUyy * input[34] * sqrt_f)) - (2.f * gammaUxz * input[33] * sqrt_f)) - (2.f * gammaUxy * input[32] * sqrt_f)) - (gammaUxx * input[31] * sqrt_f)) - (2.f * gammaUyz * input[5] * sqrt_f)) - (gammaUyy * input[4] * sqrt_f)) - (2.f * gammaUxz * input[3] * sqrt_f)) - (2.f * gammaUxy * input[2] * sqrt_f)) - (gammaUxx * input[1] * sqrt_f)) + input[0]) / (2.f * sqrt_f * gammaUzz));
		results[34] = input[27];
		results[35] = input[28];
		results[36] = input[29];
	}
}

__kernel void addSource(
	__global real* derivBuffer,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	__global real* deriv = derivBuffer + NUM_STATES * index;
	const __global real* state = stateBuffer + NUM_STATES * index;

	real alpha = state[0];
	real gamma_xx = state[1], gamma_xy = state[2], gamma_xz = state[3], gamma_yy = state[4], gamma_yz = state[5], gamma_zz = state[6];
	real A_x = state[7], A_y = state[8], A_z = state[9];
	real D_xxx = state[10], D_xxy = state[11], D_xxz = state[12], D_xyy = state[13], D_xyz = state[14], D_xzz = state[15];
	real D_yxx = state[16], D_yxy = state[17], D_yxz = state[18], D_yyy = state[19], D_yyz = state[20], D_yzz = state[21];
	real D_zxx = state[22], D_zxy = state[23], D_zxz = state[24], D_zyy = state[25], D_zyz = state[26], D_zzz = state[27];
	real K_xx = state[28], K_xy = state[29], K_xz = state[30], K_yy = state[31], K_yz = state[32], K_zz = state[33];
	real V_x = state[34], V_y = state[35], V_z = state[36];
	real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
	real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
	real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];
	real f = ADM_BONA_MASSO_F;	//could be based on alpha...


// source terms
real KUL[3][3] = {
{gammaUxx * K_xx + gammaUxy * K_xy + gammaUxz * K_xz,
gammaUxx * K_xy + gammaUxy * K_yy + gammaUxz * K_yz,
gammaUxx * K_xz + gammaUxy * K_yz + gammaUxz * K_zz,
},{gammaUxy * K_xx + gammaUyy * K_xy + gammaUyz * K_xz,
gammaUxy * K_xy + gammaUyy * K_yy + gammaUyz * K_yz,
gammaUxy * K_xz + gammaUyy * K_yz + gammaUyz * K_zz,
},{gammaUxz * K_xx + gammaUyz * K_xy + gammaUzz * K_xz,
gammaUxz * K_xy + gammaUyz * K_yy + gammaUzz * K_yz,
gammaUxz * K_xz + gammaUyz * K_yz + gammaUzz * K_zz,
},};
real trK = KUL[0][0] + KUL[1][1] + KUL[2][2];
real KSqSymLL[6] = {
K_xx * KUL[0][0] + K_xy * KUL[1][0] + K_xz * KUL[2][0],
K_xx * KUL[0][1] + K_xy * KUL[1][1] + K_xz * KUL[2][1],
K_xx * KUL[0][2] + K_xy * KUL[1][2] + K_xz * KUL[2][2],
K_xy * KUL[0][1] + K_yy * KUL[1][1] + K_yz * KUL[2][1],
K_xy * KUL[0][2] + K_yy * KUL[1][2] + K_yz * KUL[2][2],
K_xz * KUL[0][2] + K_yz * KUL[1][2] + K_zz * KUL[2][2],
};
real DLUL[3][3][3] = {
{{D_xxx * gammaUxx + D_xxy * gammaUxy + D_xxz * gammaUxz,
D_xxy * gammaUxx + D_xyy * gammaUxy + D_xyz * gammaUxz,
D_xxz * gammaUxx + D_xyz * gammaUxy + D_xzz * gammaUxz,
},{D_xxx * gammaUxy + D_xxy * gammaUyy + D_xxz * gammaUyz,
D_xxy * gammaUxy + D_xyy * gammaUyy + D_xyz * gammaUyz,
D_xxz * gammaUxy + D_xyz * gammaUyy + D_xzz * gammaUyz,
},{D_xxx * gammaUxz + D_xxy * gammaUyz + D_xxz * gammaUzz,
D_xxy * gammaUxz + D_xyy * gammaUyz + D_xyz * gammaUzz,
D_xxz * gammaUxz + D_xyz * gammaUyz + D_xzz * gammaUzz,
},},{{D_yxx * gammaUxx + D_yxy * gammaUxy + D_yxz * gammaUxz,
D_yxy * gammaUxx + D_yyy * gammaUxy + D_yyz * gammaUxz,
D_yxz * gammaUxx + D_yyz * gammaUxy + D_yzz * gammaUxz,
},{D_yxx * gammaUxy + D_yxy * gammaUyy + D_yxz * gammaUyz,
D_yxy * gammaUxy + D_yyy * gammaUyy + D_yyz * gammaUyz,
D_yxz * gammaUxy + D_yyz * gammaUyy + D_yzz * gammaUyz,
},{D_yxx * gammaUxz + D_yxy * gammaUyz + D_yxz * gammaUzz,
D_yxy * gammaUxz + D_yyy * gammaUyz + D_yyz * gammaUzz,
D_yxz * gammaUxz + D_yyz * gammaUyz + D_yzz * gammaUzz,
},},{{D_zxx * gammaUxx + D_zxy * gammaUxy + D_zxz * gammaUxz,
D_zxy * gammaUxx + D_zyy * gammaUxy + D_zyz * gammaUxz,
D_zxz * gammaUxx + D_zyz * gammaUxy + D_zzz * gammaUxz,
},{D_zxx * gammaUxy + D_zxy * gammaUyy + D_zxz * gammaUyz,
D_zxy * gammaUxy + D_zyy * gammaUyy + D_zyz * gammaUyz,
D_zxz * gammaUxy + D_zyz * gammaUyy + D_zzz * gammaUyz,
},{D_zxx * gammaUxz + D_zxy * gammaUyz + D_zxz * gammaUzz,
D_zxy * gammaUxz + D_zyy * gammaUyz + D_zyz * gammaUzz,
D_zxz * gammaUxz + D_zyz * gammaUyz + D_zzz * gammaUzz,
},},};
real D1L[3] = {
DLUL[0][0][0] + DLUL[0][1][1] + DLUL[0][2][2],
DLUL[1][0][0] + DLUL[1][1][1] + DLUL[1][2][2],
DLUL[2][0][0] + DLUL[2][1][1] + DLUL[2][2][2],
};
real D3L[3] = {
DLUL[0][0][0] + DLUL[1][1][0] + DLUL[2][2][0],
DLUL[0][0][1] + DLUL[1][1][1] + DLUL[2][2][1],
DLUL[0][0][2] + DLUL[1][1][2] + DLUL[2][2][2],
};
real DUUL[3][3][3] = {
{{DLUL[0][0][0] * gammaUxx + DLUL[1][0][0] * gammaUxy + DLUL[2][0][0] * gammaUxz,
DLUL[0][0][1] * gammaUxx + DLUL[1][0][1] * gammaUxy + DLUL[2][0][1] * gammaUxz,
DLUL[0][0][2] * gammaUxx + DLUL[1][0][2] * gammaUxy + DLUL[2][0][2] * gammaUxz,
},{DLUL[0][1][0] * gammaUxx + DLUL[1][1][0] * gammaUxy + DLUL[2][1][0] * gammaUxz,
DLUL[0][1][1] * gammaUxx + DLUL[1][1][1] * gammaUxy + DLUL[2][1][1] * gammaUxz,
DLUL[0][1][2] * gammaUxx + DLUL[1][1][2] * gammaUxy + DLUL[2][1][2] * gammaUxz,
},{DLUL[0][2][0] * gammaUxx + DLUL[1][2][0] * gammaUxy + DLUL[2][2][0] * gammaUxz,
DLUL[0][2][1] * gammaUxx + DLUL[1][2][1] * gammaUxy + DLUL[2][2][1] * gammaUxz,
DLUL[0][2][2] * gammaUxx + DLUL[1][2][2] * gammaUxy + DLUL[2][2][2] * gammaUxz,
},},{{DLUL[0][0][0] * gammaUxy + DLUL[1][0][0] * gammaUyy + DLUL[2][0][0] * gammaUyz,
DLUL[0][0][1] * gammaUxy + DLUL[1][0][1] * gammaUyy + DLUL[2][0][1] * gammaUyz,
DLUL[0][0][2] * gammaUxy + DLUL[1][0][2] * gammaUyy + DLUL[2][0][2] * gammaUyz,
},{DLUL[0][1][0] * gammaUxy + DLUL[1][1][0] * gammaUyy + DLUL[2][1][0] * gammaUyz,
DLUL[0][1][1] * gammaUxy + DLUL[1][1][1] * gammaUyy + DLUL[2][1][1] * gammaUyz,
DLUL[0][1][2] * gammaUxy + DLUL[1][1][2] * gammaUyy + DLUL[2][1][2] * gammaUyz,
},{DLUL[0][2][0] * gammaUxy + DLUL[1][2][0] * gammaUyy + DLUL[2][2][0] * gammaUyz,
DLUL[0][2][1] * gammaUxy + DLUL[1][2][1] * gammaUyy + DLUL[2][2][1] * gammaUyz,
DLUL[0][2][2] * gammaUxy + DLUL[1][2][2] * gammaUyy + DLUL[2][2][2] * gammaUyz,
},},{{DLUL[0][0][0] * gammaUxz + DLUL[1][0][0] * gammaUyz + DLUL[2][0][0] * gammaUzz,
DLUL[0][0][1] * gammaUxz + DLUL[1][0][1] * gammaUyz + DLUL[2][0][1] * gammaUzz,
DLUL[0][0][2] * gammaUxz + DLUL[1][0][2] * gammaUyz + DLUL[2][0][2] * gammaUzz,
},{DLUL[0][1][0] * gammaUxz + DLUL[1][1][0] * gammaUyz + DLUL[2][1][0] * gammaUzz,
DLUL[0][1][1] * gammaUxz + DLUL[1][1][1] * gammaUyz + DLUL[2][1][1] * gammaUzz,
DLUL[0][1][2] * gammaUxz + DLUL[1][1][2] * gammaUyz + DLUL[2][1][2] * gammaUzz,
},{DLUL[0][2][0] * gammaUxz + DLUL[1][2][0] * gammaUyz + DLUL[2][2][0] * gammaUzz,
DLUL[0][2][1] * gammaUxz + DLUL[1][2][1] * gammaUyz + DLUL[2][2][1] * gammaUzz,
DLUL[0][2][2] * gammaUxz + DLUL[1][2][2] * gammaUyz + DLUL[2][2][2] * gammaUzz,
},},};
real D12SymLL[6] = {
D_xxx * DUUL[0][0][0] + D_xxy * DUUL[0][1][0] + D_xxz * DUUL[0][2][0] + D_yxx * DUUL[1][0][0] + D_yxy * DUUL[1][1][0] + D_yxz * DUUL[1][2][0] + D_zxx * DUUL[2][0][0] + D_zxy * DUUL[2][1][0] + D_zxz * DUUL[2][2][0],
D_xxy * DUUL[0][0][0] + D_xyy * DUUL[0][1][0] + D_xyz * DUUL[0][2][0] + D_yxy * DUUL[1][0][0] + D_yyy * DUUL[1][1][0] + D_yyz * DUUL[1][2][0] + D_zxy * DUUL[2][0][0] + D_zyy * DUUL[2][1][0] + D_zyz * DUUL[2][2][0],
D_xxz * DUUL[0][0][0] + D_xyz * DUUL[0][1][0] + D_xzz * DUUL[0][2][0] + D_yxz * DUUL[1][0][0] + D_yyz * DUUL[1][1][0] + D_yzz * DUUL[1][2][0] + D_zxz * DUUL[2][0][0] + D_zyz * DUUL[2][1][0] + D_zzz * DUUL[2][2][0],
D_xxy * DUUL[0][0][1] + D_xyy * DUUL[0][1][1] + D_xyz * DUUL[0][2][1] + D_yxy * DUUL[1][0][1] + D_yyy * DUUL[1][1][1] + D_yyz * DUUL[1][2][1] + D_zxy * DUUL[2][0][1] + D_zyy * DUUL[2][1][1] + D_zyz * DUUL[2][2][1],
D_xxz * DUUL[0][0][1] + D_xyz * DUUL[0][1][1] + D_xzz * DUUL[0][2][1] + D_yxz * DUUL[1][0][1] + D_yyz * DUUL[1][1][1] + D_yzz * DUUL[1][2][1] + D_zxz * DUUL[2][0][1] + D_zyz * DUUL[2][1][1] + D_zzz * DUUL[2][2][1],
D_xxz * DUUL[0][0][2] + D_xyz * DUUL[0][1][2] + D_xzz * DUUL[0][2][2] + D_yxz * DUUL[1][0][2] + D_yyz * DUUL[1][1][2] + D_yzz * DUUL[1][2][2] + D_zxz * DUUL[2][0][2] + D_zyz * DUUL[2][1][2] + D_zzz * DUUL[2][2][2],
};
real GammaLSymLL[3][6] = {
{D_xxx,
D_yxx,
D_zxx,
((2 * D_yxy) - D_xyy),
(D_zxy + (D_yxz - D_xyz)),
((2 * D_zxz) - D_xzz),
},{((2 * D_xxy) - D_yxx),
D_xyy,
(D_zxy + (D_xyz - D_yxz)),
D_yyy,
D_zyy,
((2 * D_zyz) - D_yzz),
},{((2 * D_xxz) - D_zxx),
(D_yxz + (D_xyz - D_zxy)),
D_xzz,
((2 * D_yyz) - D_zyy),
D_yzz,
D_zzz,
},};
real GammaUSymLL[3][6] = {
{gammaUxx * GammaLSymLL[0][0] + gammaUxy * GammaLSymLL[1][0] + gammaUxz * GammaLSymLL[2][0],
gammaUxx * GammaLSymLL[0][1] + gammaUxy * GammaLSymLL[1][1] + gammaUxz * GammaLSymLL[2][1],
gammaUxx * GammaLSymLL[0][2] + gammaUxy * GammaLSymLL[1][2] + gammaUxz * GammaLSymLL[2][2],
gammaUxx * GammaLSymLL[0][3] + gammaUxy * GammaLSymLL[1][3] + gammaUxz * GammaLSymLL[2][3],
gammaUxx * GammaLSymLL[0][4] + gammaUxy * GammaLSymLL[1][4] + gammaUxz * GammaLSymLL[2][4],
gammaUxx * GammaLSymLL[0][5] + gammaUxy * GammaLSymLL[1][5] + gammaUxz * GammaLSymLL[2][5],
},{gammaUxy * GammaLSymLL[0][0] + gammaUyy * GammaLSymLL[1][0] + gammaUyz * GammaLSymLL[2][0],
gammaUxy * GammaLSymLL[0][1] + gammaUyy * GammaLSymLL[1][1] + gammaUyz * GammaLSymLL[2][1],
gammaUxy * GammaLSymLL[0][2] + gammaUyy * GammaLSymLL[1][2] + gammaUyz * GammaLSymLL[2][2],
gammaUxy * GammaLSymLL[0][3] + gammaUyy * GammaLSymLL[1][3] + gammaUyz * GammaLSymLL[2][3],
gammaUxy * GammaLSymLL[0][4] + gammaUyy * GammaLSymLL[1][4] + gammaUyz * GammaLSymLL[2][4],
gammaUxy * GammaLSymLL[0][5] + gammaUyy * GammaLSymLL[1][5] + gammaUyz * GammaLSymLL[2][5],
},{gammaUxz * GammaLSymLL[0][0] + gammaUyz * GammaLSymLL[1][0] + gammaUzz * GammaLSymLL[2][0],
gammaUxz * GammaLSymLL[0][1] + gammaUyz * GammaLSymLL[1][1] + gammaUzz * GammaLSymLL[2][1],
gammaUxz * GammaLSymLL[0][2] + gammaUyz * GammaLSymLL[1][2] + gammaUzz * GammaLSymLL[2][2],
gammaUxz * GammaLSymLL[0][3] + gammaUyz * GammaLSymLL[1][3] + gammaUzz * GammaLSymLL[2][3],
gammaUxz * GammaLSymLL[0][4] + gammaUyz * GammaLSymLL[1][4] + gammaUzz * GammaLSymLL[2][4],
gammaUxz * GammaLSymLL[0][5] + gammaUyz * GammaLSymLL[1][5] + gammaUzz * GammaLSymLL[2][5],
},};
real Gamma3L[3] = {
GammaUSymLL[0][0] + GammaUSymLL[1][1] + GammaUSymLL[2][2],
GammaUSymLL[0][1] + GammaUSymLL[1][3] + GammaUSymLL[2][4],
GammaUSymLL[0][2] + GammaUSymLL[1][4] + GammaUSymLL[2][5],
};
real Gamma31SymLL[6] = {
Gamma3L[0] * GammaUSymLL[0][0] + Gamma3L[1] * GammaUSymLL[1][0] + Gamma3L[2] * GammaUSymLL[2][0],
Gamma3L[0] * GammaUSymLL[0][1] + Gamma3L[1] * GammaUSymLL[1][1] + Gamma3L[2] * GammaUSymLL[2][1],
Gamma3L[0] * GammaUSymLL[0][2] + Gamma3L[1] * GammaUSymLL[1][2] + Gamma3L[2] * GammaUSymLL[2][2],
Gamma3L[0] * GammaUSymLL[0][3] + Gamma3L[1] * GammaUSymLL[1][3] + Gamma3L[2] * GammaUSymLL[2][3],
Gamma3L[0] * GammaUSymLL[0][4] + Gamma3L[1] * GammaUSymLL[1][4] + Gamma3L[2] * GammaUSymLL[2][4],
Gamma3L[0] * GammaUSymLL[0][5] + Gamma3L[1] * GammaUSymLL[1][5] + Gamma3L[2] * GammaUSymLL[2][5],
};
real GammaLUL[3][3][3] = {
{{gammaUxx * GammaLSymLL[0][0] + gammaUxy * GammaLSymLL[0][1] + gammaUxz * GammaLSymLL[0][2],
gammaUxx * GammaLSymLL[0][1] + gammaUxy * GammaLSymLL[0][3] + gammaUxz * GammaLSymLL[0][4],
gammaUxx * GammaLSymLL[0][2] + gammaUxy * GammaLSymLL[0][4] + gammaUxz * GammaLSymLL[0][5],
},{gammaUxy * GammaLSymLL[0][0] + gammaUyy * GammaLSymLL[0][1] + gammaUyz * GammaLSymLL[0][2],
gammaUxy * GammaLSymLL[0][1] + gammaUyy * GammaLSymLL[0][3] + gammaUyz * GammaLSymLL[0][4],
gammaUxy * GammaLSymLL[0][2] + gammaUyy * GammaLSymLL[0][4] + gammaUyz * GammaLSymLL[0][5],
},{gammaUxz * GammaLSymLL[0][0] + gammaUyz * GammaLSymLL[0][1] + gammaUzz * GammaLSymLL[0][2],
gammaUxz * GammaLSymLL[0][1] + gammaUyz * GammaLSymLL[0][3] + gammaUzz * GammaLSymLL[0][4],
gammaUxz * GammaLSymLL[0][2] + gammaUyz * GammaLSymLL[0][4] + gammaUzz * GammaLSymLL[0][5],
},},{{gammaUxx * GammaLSymLL[1][0] + gammaUxy * GammaLSymLL[1][1] + gammaUxz * GammaLSymLL[1][2],
gammaUxx * GammaLSymLL[1][1] + gammaUxy * GammaLSymLL[1][3] + gammaUxz * GammaLSymLL[1][4],
gammaUxx * GammaLSymLL[1][2] + gammaUxy * GammaLSymLL[1][4] + gammaUxz * GammaLSymLL[1][5],
},{gammaUxy * GammaLSymLL[1][0] + gammaUyy * GammaLSymLL[1][1] + gammaUyz * GammaLSymLL[1][2],
gammaUxy * GammaLSymLL[1][1] + gammaUyy * GammaLSymLL[1][3] + gammaUyz * GammaLSymLL[1][4],
gammaUxy * GammaLSymLL[1][2] + gammaUyy * GammaLSymLL[1][4] + gammaUyz * GammaLSymLL[1][5],
},{gammaUxz * GammaLSymLL[1][0] + gammaUyz * GammaLSymLL[1][1] + gammaUzz * GammaLSymLL[1][2],
gammaUxz * GammaLSymLL[1][1] + gammaUyz * GammaLSymLL[1][3] + gammaUzz * GammaLSymLL[1][4],
gammaUxz * GammaLSymLL[1][2] + gammaUyz * GammaLSymLL[1][4] + gammaUzz * GammaLSymLL[1][5],
},},{{gammaUxx * GammaLSymLL[2][0] + gammaUxy * GammaLSymLL[2][1] + gammaUxz * GammaLSymLL[2][2],
gammaUxx * GammaLSymLL[2][1] + gammaUxy * GammaLSymLL[2][3] + gammaUxz * GammaLSymLL[2][4],
gammaUxx * GammaLSymLL[2][2] + gammaUxy * GammaLSymLL[2][4] + gammaUxz * GammaLSymLL[2][5],
},{gammaUxy * GammaLSymLL[2][0] + gammaUyy * GammaLSymLL[2][1] + gammaUyz * GammaLSymLL[2][2],
gammaUxy * GammaLSymLL[2][1] + gammaUyy * GammaLSymLL[2][3] + gammaUyz * GammaLSymLL[2][4],
gammaUxy * GammaLSymLL[2][2] + gammaUyy * GammaLSymLL[2][4] + gammaUyz * GammaLSymLL[2][5],
},{gammaUxz * GammaLSymLL[2][0] + gammaUyz * GammaLSymLL[2][1] + gammaUzz * GammaLSymLL[2][2],
gammaUxz * GammaLSymLL[2][1] + gammaUyz * GammaLSymLL[2][3] + gammaUzz * GammaLSymLL[2][4],
gammaUxz * GammaLSymLL[2][2] + gammaUyz * GammaLSymLL[2][4] + gammaUzz * GammaLSymLL[2][5],
},},};
real GammaLSymUU[3][6] = {
{gammaUxx * GammaLUL[0][0][0] + gammaUxy * GammaLUL[0][0][1] + gammaUxz * GammaLUL[0][0][2],
gammaUxy * GammaLUL[0][0][0] + gammaUyy * GammaLUL[0][0][1] + gammaUyz * GammaLUL[0][0][2],
gammaUxz * GammaLUL[0][0][0] + gammaUyz * GammaLUL[0][0][1] + gammaUzz * GammaLUL[0][0][2],
gammaUxy * GammaLUL[0][1][0] + gammaUyy * GammaLUL[0][1][1] + gammaUyz * GammaLUL[0][1][2],
gammaUxz * GammaLUL[0][1][0] + gammaUyz * GammaLUL[0][1][1] + gammaUzz * GammaLUL[0][1][2],
gammaUxz * GammaLUL[0][2][0] + gammaUyz * GammaLUL[0][2][1] + gammaUzz * GammaLUL[0][2][2],
},{gammaUxx * GammaLUL[1][0][0] + gammaUxy * GammaLUL[1][0][1] + gammaUxz * GammaLUL[1][0][2],
gammaUxy * GammaLUL[1][0][0] + gammaUyy * GammaLUL[1][0][1] + gammaUyz * GammaLUL[1][0][2],
gammaUxz * GammaLUL[1][0][0] + gammaUyz * GammaLUL[1][0][1] + gammaUzz * GammaLUL[1][0][2],
gammaUxy * GammaLUL[1][1][0] + gammaUyy * GammaLUL[1][1][1] + gammaUyz * GammaLUL[1][1][2],
gammaUxz * GammaLUL[1][1][0] + gammaUyz * GammaLUL[1][1][1] + gammaUzz * GammaLUL[1][1][2],
gammaUxz * GammaLUL[1][2][0] + gammaUyz * GammaLUL[1][2][1] + gammaUzz * GammaLUL[1][2][2],
},{gammaUxx * GammaLUL[2][0][0] + gammaUxy * GammaLUL[2][0][1] + gammaUxz * GammaLUL[2][0][2],
gammaUxy * GammaLUL[2][0][0] + gammaUyy * GammaLUL[2][0][1] + gammaUyz * GammaLUL[2][0][2],
gammaUxz * GammaLUL[2][0][0] + gammaUyz * GammaLUL[2][0][1] + gammaUzz * GammaLUL[2][0][2],
gammaUxy * GammaLUL[2][1][0] + gammaUyy * GammaLUL[2][1][1] + gammaUyz * GammaLUL[2][1][2],
gammaUxz * GammaLUL[2][1][0] + gammaUyz * GammaLUL[2][1][1] + gammaUzz * GammaLUL[2][1][2],
gammaUxz * GammaLUL[2][2][0] + gammaUyz * GammaLUL[2][2][1] + gammaUzz * GammaLUL[2][2][2],
},};
real Gamma11SymLL[6] = {
GammaLSymLL[0][0] * GammaLSymUU[0][0] + GammaLSymLL[0][1] * GammaLSymUU[0][1] + GammaLSymLL[0][2] * GammaLSymUU[0][2] + GammaLSymLL[0][1] * GammaLSymUU[0][1] + GammaLSymLL[0][3] * GammaLSymUU[0][3] + GammaLSymLL[0][4] * GammaLSymUU[0][4] + GammaLSymLL[0][2] * GammaLSymUU[0][2] + GammaLSymLL[0][4] * GammaLSymUU[0][4] + GammaLSymLL[0][5] * GammaLSymUU[0][5],
GammaLSymLL[0][0] * GammaLSymUU[1][0] + GammaLSymLL[0][1] * GammaLSymUU[1][1] + GammaLSymLL[0][2] * GammaLSymUU[1][2] + GammaLSymLL[0][1] * GammaLSymUU[1][1] + GammaLSymLL[0][3] * GammaLSymUU[1][3] + GammaLSymLL[0][4] * GammaLSymUU[1][4] + GammaLSymLL[0][2] * GammaLSymUU[1][2] + GammaLSymLL[0][4] * GammaLSymUU[1][4] + GammaLSymLL[0][5] * GammaLSymUU[1][5],
GammaLSymLL[0][0] * GammaLSymUU[2][0] + GammaLSymLL[0][1] * GammaLSymUU[2][1] + GammaLSymLL[0][2] * GammaLSymUU[2][2] + GammaLSymLL[0][1] * GammaLSymUU[2][1] + GammaLSymLL[0][3] * GammaLSymUU[2][3] + GammaLSymLL[0][4] * GammaLSymUU[2][4] + GammaLSymLL[0][2] * GammaLSymUU[2][2] + GammaLSymLL[0][4] * GammaLSymUU[2][4] + GammaLSymLL[0][5] * GammaLSymUU[2][5],
GammaLSymLL[1][0] * GammaLSymUU[1][0] + GammaLSymLL[1][1] * GammaLSymUU[1][1] + GammaLSymLL[1][2] * GammaLSymUU[1][2] + GammaLSymLL[1][1] * GammaLSymUU[1][1] + GammaLSymLL[1][3] * GammaLSymUU[1][3] + GammaLSymLL[1][4] * GammaLSymUU[1][4] + GammaLSymLL[1][2] * GammaLSymUU[1][2] + GammaLSymLL[1][4] * GammaLSymUU[1][4] + GammaLSymLL[1][5] * GammaLSymUU[1][5],
GammaLSymLL[1][0] * GammaLSymUU[2][0] + GammaLSymLL[1][1] * GammaLSymUU[2][1] + GammaLSymLL[1][2] * GammaLSymUU[2][2] + GammaLSymLL[1][1] * GammaLSymUU[2][1] + GammaLSymLL[1][3] * GammaLSymUU[2][3] + GammaLSymLL[1][4] * GammaLSymUU[2][4] + GammaLSymLL[1][2] * GammaLSymUU[2][2] + GammaLSymLL[1][4] * GammaLSymUU[2][4] + GammaLSymLL[1][5] * GammaLSymUU[2][5],
GammaLSymLL[2][0] * GammaLSymUU[2][0] + GammaLSymLL[2][1] * GammaLSymUU[2][1] + GammaLSymLL[2][2] * GammaLSymUU[2][2] + GammaLSymLL[2][1] * GammaLSymUU[2][1] + GammaLSymLL[2][3] * GammaLSymUU[2][3] + GammaLSymLL[2][4] * GammaLSymUU[2][4] + GammaLSymLL[2][2] * GammaLSymUU[2][2] + GammaLSymLL[2][4] * GammaLSymUU[2][4] + GammaLSymLL[2][5] * GammaLSymUU[2][5],
};
real ADL[3] = {
A_x - 2 * D3L[0],
A_y - 2 * D3L[1],
A_z - 2 * D3L[2],
};
real ADU[3] = {
gammaUxx * ADL[0] + gammaUxy * ADL[1] + gammaUxz * ADL[2],
gammaUxy * ADL[0] + gammaUyy * ADL[1] + gammaUyz * ADL[2],
gammaUxz * ADL[0] + gammaUyz * ADL[1] + gammaUzz * ADL[2],
};
real ADDSymLL[6] = {
ADU[0] * (2 * D_xxx) + ADU[1] * (2 * D_xxy) + ADU[2] * (2 * D_xxz),
ADU[0] * (D_xxy + D_yxx) + ADU[1] * (D_xyy + D_yxy) + ADU[2] * (D_xyz + D_yxz),
ADU[0] * (D_xxz + D_zxx) + ADU[1] * (D_xyz + D_zxy) + ADU[2] * (D_xzz + D_zxz),
ADU[0] * (2 * D_yxy) + ADU[1] * (2 * D_yyy) + ADU[2] * (2 * D_yyz),
ADU[0] * (D_yxz + D_zxy) + ADU[1] * (D_yyz + D_zyy) + ADU[2] * (D_yzz + D_zyz),
ADU[0] * (2 * D_zxz) + ADU[1] * (2 * D_zyz) + ADU[2] * (2 * D_zzz),
};
real R4SymLL[6] = {
0,
0,
0,
0,
0,
0,
};
real SSymLL[6] = {
-R4SymLL[0] + trK * K_xx - 2 * KSqSymLL[0] + 4 * D12SymLL[0] + Gamma31SymLL[0] - Gamma11SymLL[0] + ADDSymLL[0] + (A_x * ((2 * V_x) - D1L[0])),
-R4SymLL[1] + trK * K_xy - 2 * KSqSymLL[1] + 4 * D12SymLL[1] + Gamma31SymLL[1] - Gamma11SymLL[1] + ADDSymLL[1] + ((((2 * A_y * V_x) - (A_y * D1L[0])) + ((2 * A_x * V_y) - (A_x * D1L[1]))) / 2),
-R4SymLL[2] + trK * K_xz - 2 * KSqSymLL[2] + 4 * D12SymLL[2] + Gamma31SymLL[2] - Gamma11SymLL[2] + ADDSymLL[2] + ((((2 * A_z * V_x) - (A_z * D1L[0])) + ((2 * A_x * V_z) - (A_x * D1L[2]))) / 2),
-R4SymLL[3] + trK * K_yy - 2 * KSqSymLL[3] + 4 * D12SymLL[3] + Gamma31SymLL[3] - Gamma11SymLL[3] + ADDSymLL[3] + (A_y * ((2 * V_y) - D1L[1])),
-R4SymLL[4] + trK * K_yz - 2 * KSqSymLL[4] + 4 * D12SymLL[4] + Gamma31SymLL[4] - Gamma11SymLL[4] + ADDSymLL[4] + ((((2 * A_z * V_y) - (A_z * D1L[1])) + ((2 * A_y * V_z) - (A_y * D1L[2]))) / 2),
-R4SymLL[5] + trK * K_zz - 2 * KSqSymLL[5] + 4 * D12SymLL[5] + Gamma31SymLL[5] - Gamma11SymLL[5] + ADDSymLL[5] + (A_z * ((2 * V_z) - D1L[2])),
};
real GU0L[3] = {
0,
0,
0,
};
real AKL[3] = {
A_x * KUL[0][0] + A_y * KUL[1][0] + A_z * KUL[2][0],
A_x * KUL[0][1] + A_y * KUL[1][1] + A_z * KUL[2][1],
A_x * KUL[0][2] + A_y * KUL[1][2] + A_z * KUL[2][2],
};
real K12D23L[3] = {
KUL[0][0] * DLUL[0][0][0] +KUL[0][1] * DLUL[0][1][0] +KUL[0][2] * DLUL[0][2][0] + KUL[1][0] * DLUL[0][0][1] +KUL[1][1] * DLUL[0][1][1] +KUL[1][2] * DLUL[0][2][1] + KUL[2][0] * DLUL[0][0][2] +KUL[2][1] * DLUL[0][1][2] +KUL[2][2] * DLUL[0][2][2],
KUL[0][0] * DLUL[1][0][0] +KUL[0][1] * DLUL[1][1][0] +KUL[0][2] * DLUL[1][2][0] + KUL[1][0] * DLUL[1][0][1] +KUL[1][1] * DLUL[1][1][1] +KUL[1][2] * DLUL[1][2][1] + KUL[2][0] * DLUL[1][0][2] +KUL[2][1] * DLUL[1][1][2] +KUL[2][2] * DLUL[1][2][2],
KUL[0][0] * DLUL[2][0][0] +KUL[0][1] * DLUL[2][1][0] +KUL[0][2] * DLUL[2][2][0] + KUL[1][0] * DLUL[2][0][1] +KUL[1][1] * DLUL[2][1][1] +KUL[1][2] * DLUL[2][2][1] + KUL[2][0] * DLUL[2][0][2] +KUL[2][1] * DLUL[2][1][2] +KUL[2][2] * DLUL[2][2][2],
};
real KD23L[3] = {
KUL[0][0] * D1L[0] + KUL[1][0] * D1L[1] + KUL[2][0] * D1L[2],
KUL[0][1] * D1L[0] + KUL[1][1] * D1L[1] + KUL[2][1] * D1L[2],
KUL[0][2] * D1L[0] + KUL[1][2] * D1L[1] + KUL[2][2] * D1L[2],
};
real K12D12L[3] = {
KUL[0][0] * DLUL[0][0][0] + KUL[0][1] * DLUL[0][1][0] + KUL[0][2] * DLUL[0][2][0] + KUL[1][0] * DLUL[1][0][0] + KUL[1][1] * DLUL[1][1][0] + KUL[1][2] * DLUL[1][2][0] + KUL[2][0] * DLUL[2][0][0] + KUL[2][1] * DLUL[2][1][0] + KUL[2][2] * DLUL[2][2][0],
KUL[0][0] * DLUL[0][0][1] + KUL[0][1] * DLUL[0][1][1] + KUL[0][2] * DLUL[0][2][1] + KUL[1][0] * DLUL[1][0][1] + KUL[1][1] * DLUL[1][1][1] + KUL[1][2] * DLUL[1][2][1] + KUL[2][0] * DLUL[2][0][1] + KUL[2][1] * DLUL[2][1][1] + KUL[2][2] * DLUL[2][2][1],
KUL[0][0] * DLUL[0][0][2] + KUL[0][1] * DLUL[0][1][2] + KUL[0][2] * DLUL[0][2][2] + KUL[1][0] * DLUL[1][0][2] + KUL[1][1] * DLUL[1][1][2] + KUL[1][2] * DLUL[1][2][2] + KUL[2][0] * DLUL[2][0][2] + KUL[2][1] * DLUL[2][1][2] + KUL[2][2] * DLUL[2][2][2],
};
real KD12L[3] = {
KUL[0][0] * D3L[0] + KUL[1][0] * D3L[1] + KUL[2][0] * D3L[2],
KUL[0][1] * D3L[0] + KUL[1][1] * D3L[1] + KUL[2][1] * D3L[2],
KUL[0][2] * D3L[0] + KUL[1][2] * D3L[1] + KUL[2][2] * D3L[2],
};
real PL[3] = {
GU0L[0] + AKL[0] - A_x * trK + K12D23L[0] + KD23L[0] - 2 * K12D12L[0] + 2 * KD12L[0],
GU0L[1] + AKL[1] - A_y * trK + K12D23L[1] + KD23L[1] - 2 * K12D12L[1] + 2 * KD12L[1],
GU0L[2] + AKL[2] - A_z * trK + K12D23L[2] + KD23L[2] - 2 * K12D12L[2] + 2 * KD12L[2],
};


	deriv[0] += -alpha * alpha * f * trK;
	deriv[1] += -2.f * alpha * K_xx;
	deriv[2] += -2.f * alpha * K_xy;
	deriv[3] += -2.f * alpha * K_xz;
	deriv[4] += -2.f * alpha * K_yy;
	deriv[5] += -2.f * alpha * K_yz;
	deriv[6] += -2.f * alpha * K_zz;
	deriv[28] += alpha * SSymLL[0];
	deriv[29] += alpha * SSymLL[1];
	deriv[30] += alpha * SSymLL[2];
	deriv[31] += alpha * SSymLL[3];
	deriv[32] += alpha * SSymLL[4];
	deriv[33] += alpha * SSymLL[5];
	deriv[34] += alpha * PL[0];
	deriv[35] += alpha * PL[1];
	deriv[36] += alpha * PL[2];


}

// the 1D version has no problems, but at 2D we get instabilities ... 
__kernel void constrain(
	__global real* stateBuffer)
{
#if 0
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	__global real* state = stateBuffer + NUM_STATES * index;

	//real alpha = state[0];
	real gamma_xx = state[1], gamma_xy = state[2], gamma_xz = state[3], gamma_yy = state[4], gamma_yz = state[5], gamma_zz = state[6];
	//real A_x = state[7], A_y = state[8], A_z = state[9];
	real /*D_xxx = state[10],*/ D_xxy = state[11], D_xxz = state[12], D_xyy = state[13], D_xyz = state[14], D_xzz = state[15];
	real D_yxx = state[16], D_yxy = state[17], D_yxz = state[18], /*D_yyy = state[19],*/ D_yyz = state[20], D_yzz = state[21];
	real D_zxx = state[22], D_zxy = state[23], D_zxz = state[24], D_zyy = state[25], D_zyz = state[26]/*, D_zzz = state[27]*/;
	//real K_xx = state[28], K_xy = state[29], K_xz = state[30], K_yy = state[31], K_yz = state[32], K_zz = state[33];
	//real V_x = state[34], V_y = state[35], V_z = state[36];
	real gamma = det3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz);
	real8 gammaInv = inv3x3sym(gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz, gamma);
	real gammaUxx = gammaInv[0], gammaUxy = gammaInv[1], gammaUxz = gammaInv[2], gammaUyy = gammaInv[3], gammaUyz = gammaInv[4], gammaUzz = gammaInv[5];

	real D3_D1_x = 
		(gammaUxy * D_xxy)
		+ (gammaUxz * D_xxz)
		+ (gammaUyy * D_xyy)
		+ (2. * gammaUyz * D_xyz)
		+ (gammaUzz * D_xzz)
		- (gammaUxy * D_yxx)
		- (gammaUxz * D_zxx)
		- (gammaUyy * D_yxy)
		- (gammaUyz * D_zxy)
		- (gammaUyz * D_yxz)
		- (gammaUzz * D_zxz);
	real D3_D1_y = 
		(gammaUxx * D_yxx)
		+ (gammaUxy * D_yxy)
		+ (2. * gammaUxz * D_yxz)
		+ (gammaUyz * D_yyz)
		+ (gammaUzz * D_yzz)
		- (gammaUxx * D_xxy)
		- (gammaUxz * D_zxy)
		- (gammaUxy * D_xyy)
		- (gammaUyz * D_zyy)
		- (gammaUxz * D_xyz)
		- (gammaUzz * D_zyz);
	real D3_D1_z = 
		(gammaUxx * D_zxx)
		+ (2. * gammaUxy * D_zxy)
		+ (gammaUxz * D_zxz)
		+ (gammaUyy * D_zyy)
		+ (gammaUyz * D_zyz)
		- (gammaUxx * D_xxz)
		- (gammaUxy * D_yxz)
		- (gammaUxy * D_xyz)
		- (gammaUyy * D_yyz)
		- (gammaUxz * D_xzz)
		- (gammaUyz * D_yzz);

#if 0	//directly assign V_i's
	state[34] = D3_D1_x;
	state[35] = D3_D1_y;
	state[36] = D3_D1_z;
#endif
#if 1	//linearly project out the [V_i, D_ijk] vector

#endif

#endif
}

/*
The components of the Roe solver specific to the Euler equations
paritcularly the spectral decomposition
*/

//#ifdef AMD_SUCKS...
//#pragma OPENCL EXTENSION cl_amd_printf : enable

#include "HydroGPU/Shared/Common.h"

#define gamma idealGas_heatCapacityRatio	//laziness

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* primitiveBuffer,
	int side);

//From Marti & Muller 2008
void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,	//not used atm. TODO add Roe averaging, and use this.
	const __global real* primitiveBuffer,
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

//	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
//	const __global real* stateR = stateBuffer + NUM_STATES * index;
	const __global real* primitiveL = primitiveBuffer + NUM_PRIMITIVE * indexPrev;
	const __global real* primitiveR = primitiveBuffer + NUM_PRIMITIVE * index;
	
	int interfaceIndex = side + DIM * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* evl = eigenvectorsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;
	__global real* evr = evl + NUM_STATES * NUM_STATES;

	real rhoL = primitiveL[PRIMITIVE_DENSITY];
	real4 vL = (real4)(0., 0., 0., 0.);
	vL.x = primitiveL[PRIMITIVE_VELOCITY_X];
#if DIM > 1
	vL.y = primitiveL[PRIMITIVE_VELOCITY_Y];
#if DIM > 2
	vL.z = primitiveL[PRIMITIVE_VELOCITY_Z];
#endif
#endif
	real eIntL = primitiveL[PRIMITIVE_SPECIFIC_INTERNAL_ENERGY];
	
	real rhoR = primitiveR[PRIMITIVE_DENSITY];
	real4 vR = (real4)(0., 0., 0., 0.);
	vR.x = primitiveR[PRIMITIVE_VELOCITY_X];
#if DIM > 1
	vR.y = primitiveR[PRIMITIVE_VELOCITY_Y];
#if DIM > 2
	vR.z = primitiveR[PRIMITIVE_VELOCITY_Z];
#endif
#endif
	real eIntR = primitiveR[PRIMITIVE_SPECIFIC_INTERNAL_ENERGY];

//printf("cell %d prims L=%f %f %f R=%f %f %f\n", index, rhoL, vL.x, eIntL, rhoR, vR.x, eIntR);

	real rho = .5 * (rhoL + rhoR);
	real eInt = .5 * (eIntL + eIntR);
	real4 v = .5 * (vL + vR);
	real vSq = dot(v,v);
	real oneOverW2 = 1. - vSq;
	real oneOverW = sqrt(oneOverW2);
	real W = 1. / oneOverW;
	real W2 = 1. / oneOverW2;
	real P = (gamma - 1.) * rho * eInt;
	real h = 1. + eInt + P / rho;
	real P_over_rho_h = P / (rho * h);

#if DIM > 1
	if (side == 1) {
		v = (real4)(v.y, -v.x, v.z, 0.);	// -90' rotation to put the y axis contents into the x axis
	} 
#if DIM > 2
	else if (side == 2) {
		v = (real4)(v.z, v.y, -v.x, 0.);	//-90' rotation to put the z axis in the x axis
	}
#endif
#endif

	real hW = h * W;
	real hSq = h * h;

	real vxSq = v.x * v.x;
	real csSq = gamma * P_over_rho_h;
	real cs = sqrt(csSq);

	real discr = sqrt((1. - vSq) * ((1. - vSq * csSq) - vxSq * (1. - csSq)));
	real lambdaMin = (v.x * (1. - csSq) - cs * discr) / (1. - vSq * csSq);
	real lambdaMax = (v.x * (1. - csSq) + cs * discr) / (1. - vSq * csSq);

	eigenvalues[0] = lambdaMin;
	eigenvalues[1] = v.x;
#if DIM > 1
	eigenvalues[2] = v.x;
#if DIM > 2
	eigenvalues[3] = v.x;
#endif
#endif
	eigenvalues[DIM+1] = lambdaMax;

//printf("cell %d eigenvalues %f %f %f\n", index, eigenvalues[0], eigenvalues[1], eigenvalues[DIM+1]);

	//eigenvectors

	real Kappa = h;	//true for ideal gas. otherwise the general equation for Kappa gets instable at high Lorentz factors 
	real AMinus = (1. - vxSq) / (1. - v.x * lambdaMin);
	real APlus  = (1. - vxSq) / (1. - v.x * lambdaMax);
//printf("cell %d A+ %f A- %f\n", index, APlus, AMinus);
//printf("cell %d h %f W %f hW %f\n", index, h, W, hW);
	//min col 
	evr[0 + NUM_STATES * 0] = 1.;
	evr[1 + NUM_STATES * 0] = hW * AMinus * lambdaMin;	//inf
#if DIM > 1
	evr[2 + NUM_STATES * 0] = hW * v.y;
#if DIM > 2
	evr[3 + NUM_STATES * 0] = hW * v.z;
#endif
#endif
	evr[(DIM+1) + NUM_STATES * 0] = hW * AMinus - 1.;
	//mid col (normal)
	evr[0 + NUM_STATES * 1] = oneOverW;	// = Kappa / hW
	evr[1 + NUM_STATES * 1] = v.x;
#if DIM > 1
	evr[2 + NUM_STATES * 1] = v.y;
#if DIM > 2
	evr[3 + NUM_STATES * 1] = v.z;
#endif
#endif
	evr[(DIM+1) + NUM_STATES * 1] = 1. - oneOverW;	// = 1. - Kappa / hW;
	//mid col (tangent A)
#if DIM > 1
	evr[0 + NUM_STATES * 2] = W * v.y;
	evr[1 + NUM_STATES * 2] = 2. * h * W2 * v.x * v.y;
	evr[2 + NUM_STATES * 2] = h * (1. + 2. * W2 * v.y * v.y);
#if DIM > 2
	evr[3 + NUM_STATES * 2] = 2. * h * W2 * v.y * v.z;
#endif
	evr[(DIM+1) + NUM_STATES * 2] = (2. * hW - 1.) * W * v.y;
#endif
	//mid col (tangent B)
#if DIM > 2
	evr[0 + NUM_STATES * 3] = W * v.z;
	evr[1 + NUM_STATES * 3] = 2. * h * W2 * v.x * v.z;
	evr[2 + NUM_STATES * 3] = 2. * h * W2 * v.y * v.z;
	evr[3 + NUM_STATES * 3] = h * (1. + 2. * W2 * v.z * v.z);
	evr[(DIM+1) + NUM_STATES * 3] = (2. * hW - 1.) * W * v.z;
#endif
	//max col 
	evr[0 + NUM_STATES * (DIM+1)] = 1.;
	evr[1 + NUM_STATES * (DIM+1)] = hW * APlus * lambdaMax;	//inf
#if DIM > 1
	evr[2 + NUM_STATES * (DIM+1)] = hW * v.y;
#if DIM > 2
	evr[3 + NUM_STATES * (DIM+1)] = hW * v.z;
#endif
#endif
	evr[(DIM+1) + NUM_STATES * (DIM+1)] = hW * APlus - 1.;

//for (int j = 0; j < DIM+2; ++j) {
//	for (int k = 0; k < DIM+2; ++k) {
//		printf("cell %d right eigenvectors %d %d = %f\n", index, j, k, evr[j + NUM_STATES * k]);
//	}
//}

#if 1 || DIM != 1
	//calculate eigenvector inverses ... 
	real Delta = hSq * hW * (Kappa - 1.) * (1. - vxSq) * (APlus * lambdaMax - AMinus * lambdaMin);
	
	//min row
	real scale;
	scale = hSq / Delta;
	evl[0 + NUM_STATES * 0] = scale * (hW * APlus * (v.x - lambdaMax) - v.x - W2 * (vSq - vxSq) * (2. * Kappa - 1.) * (v.x - APlus * lambdaMax) + Kappa * APlus * lambdaMax);
	evl[0 + NUM_STATES * 1] = scale * (1. + W2 * (vSq - vxSq) * (2. * Kappa - 1.) * (1. - APlus) - Kappa * APlus);
#if DIM > 1
	evl[0 + NUM_STATES * 2] = scale * (W2 * v.y * (2. * Kappa - 1.) * APlus * (v.x - lambdaMax));
#if DIM > 2
	evl[0 + NUM_STATES * 3] = scale * (W2 * v.z * (2. * Kappa - 1.) * APlus * (v.x - lambdaMax));
#endif
#endif
	evl[0 + NUM_STATES * (DIM+1)] = scale * (-v.x - W2 * (vSq - vxSq) * (2. * Kappa - 1.) * (v.x - APlus * lambdaMax) + Kappa * APlus * lambdaMax);
	//mid normal row
	scale = W / (Kappa - 1.);
	evl[1 + NUM_STATES * 0] = scale * (h - W);
	evl[1 + NUM_STATES * 1] = scale * (W * v.x);
#if DIM > 1
	evl[1 + NUM_STATES * 2] = scale * (W * v.y);
#if DIM > 2
	evl[1 + NUM_STATES * 3] = scale * (W * v.z);
#endif
#endif
	evl[1 + NUM_STATES * (DIM+1)] = scale * (-W);
	//mid tangent A row
#if DIM > 1
	scale = 1. / (h * (1. - vxSq));
	evl[2 + NUM_STATES * 0] = scale * (-v.y);
	evl[2 + NUM_STATES * 1] = scale * (v.x * v.y);
	evl[2 + NUM_STATES * 2] = scale * (1. - vxSq);
#if DIM > 2
	evl[2 + NUM_STATES * 3] = 0.;
#endif
	evl[2 + NUM_STATES * (DIM+1)] = scale * (-v.y);
#endif
	//mid tangent B row
#if DIM > 2
	evl[3 + NUM_STATES * 0] = scale * (-v.z);
	evl[3 + NUM_STATES * 1] = scale * (v.x * v.z);
	evl[3 + NUM_STATES * 2] = 0.;
	evl[3 + NUM_STATES * 3] = scale * (1. - vxSq);
	evl[3 + NUM_STATES * (DIM+1)] = scale * (-v.z);
#endif
	//max row
	scale = -hSq / Delta;
	evl[(DIM+1) + NUM_STATES * 0] = scale * (hW * AMinus * (v.x - lambdaMin) - v.x - W2 * (vSq - vxSq) * (2. * Kappa - 1.) * (v.x - AMinus * lambdaMin) + Kappa * AMinus * lambdaMin);
	evl[(DIM+1) + NUM_STATES * 1] = scale * (1. + W2 * (vSq - vxSq) * (2. * Kappa - 1.) * (1. - AMinus) - Kappa * AMinus);
#if DIM > 1
	evl[(DIM+1) + NUM_STATES * 2] = scale * (W2 * v.y * (2. * Kappa - 1.) * AMinus * (v.x - lambdaMin));
#if DIM > 2
	evl[(DIM+1) + NUM_STATES * 3] = scale * (W2 * v.z * (2. * Kappa - 1.) * AMinus * (v.x - lambdaMin));
#endif
#endif
	evl[(DIM+1) + NUM_STATES * (DIM+1)] = scale * (-v.x - W2 * (vSq - vxSq) * (2. * Kappa - 1.) * (v.x - AMinus * lambdaMin) + Kappa * AMinus * lambdaMin);
#else	//numeric inverse
	real det = evr[0 + 3 * 0] * evr[1 + 3 * 1] * evr[2 + 3 * 2]
			+ evr[1 + 3 * 0] * evr[2 + 3 * 1] * evr[0 + 3 * 2]
			+ evr[2 + 3 * 0] * evr[0 + 3 * 1] * evr[1 + 3 * 2]
			- evr[2 + 3 * 0] * evr[1 + 3 * 1] * evr[0 + 3 * 2]
			- evr[1 + 3 * 0] * evr[0 + 3 * 1] * evr[2 + 3 * 2]
			- evr[0 + 3 * 0] * evr[2 + 3 * 1] * evr[1 + 3 * 2];
	real invDet = 1. / det;
	for (int j = 0; j < 3; ++j) {                                                     
		int j1 = j % 3;
		int j2 = j1 % 3;
		for (int k = 0; k < 3; ++k) {                                                  
			int k1 = k % 3;
			int k2 = k1 % 3;
			evl[k + 3 * j] = invDet * (evr[j1 + 3 * k1] * evr[j2 + 3 * k2] - evr[j1 + 3 * k2] * evr[j2 + 3 * k1]);
		}
	}
#endif

//for (int j = 0; j < DIM+2; ++j) {
//	for (int k = 0; k < DIM+2; ++k) {
//		printf("cell %d left eigenvectors %d %d = %f\n", index, j, k, evl[j + NUM_STATES * k]);
//	}
//}

#if DIM > 1
	if (side == 1) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			//-90' rotation applied to the LHS of incoming velocity vectors, to move their y axis into the x axis
			// is equivalent of a -90' rotation applied to the RHS of the flux jacobian A
			// and A = Q V Q-1 for Q = the right eigenvectors and Q-1 the left eigenvectors
			// so a -90' rotation applied to the RHS of A is a +90' rotation applied to the RHS of Q-1 the left eigenvectors
			//and while a rotation applied to the LHS of a vector rotates the elements of its column vectors, a rotation applied to the RHS rotates the elements of its row vectors 
			//each row's y <- x, x <- -y
			tmp = evl[i + NUM_STATES * STATE_MOMENTUM_DENSITY_X];
			evl[i + NUM_STATES * STATE_MOMENTUM_DENSITY_X] = -evl[i + NUM_STATES * STATE_MOMENTUM_DENSITY_Y];
			evl[i + NUM_STATES * STATE_MOMENTUM_DENSITY_Y] = tmp;
			//a -90' rotation applied to the RHS of A must be corrected with a 90' rotation on the LHS of A
			//this rotates the elements of the column vectors by 90'
			//each column's x <- y, y <- -x
			tmp = evr[STATE_MOMENTUM_DENSITY_X + NUM_STATES * i];
			evr[STATE_MOMENTUM_DENSITY_X + NUM_STATES * i] = -evr[STATE_MOMENTUM_DENSITY_Y + NUM_STATES * i];
			evr[STATE_MOMENTUM_DENSITY_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			tmp = evl[i + NUM_STATES * STATE_MOMENTUM_DENSITY_X];
			evl[i + NUM_STATES * STATE_MOMENTUM_DENSITY_X] = -evl[i + NUM_STATES * STATE_MOMENTUM_DENSITY_Z];
			evl[i + NUM_STATES * STATE_MOMENTUM_DENSITY_Z] = tmp;
			tmp = evr[STATE_MOMENTUM_DENSITY_X + NUM_STATES * i];
			evr[STATE_MOMENTUM_DENSITY_X + NUM_STATES * i] = -evr[STATE_MOMENTUM_DENSITY_Z + NUM_STATES * i];
			evr[STATE_MOMENTUM_DENSITY_Z + NUM_STATES * i] = tmp;
		}
	}
#endif
#endif
}

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* primitiveBuffer)
{
	for (int side = 0; side < DIM; ++side) {
		calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, stateBuffer, primitiveBuffer, side);
	}
}

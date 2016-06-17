/*
using https://arxiv.org/pdf/0804.0402v1.pdf
*/

#include "HydroGPU/Shared/Common.h"

#define gamma idealGas_heatCapacityRatio	//laziness
#define mu0 mhd_vacuumPermeability
#define sqrt_mu0 mhd_sqrt_vacuumPermeability

#if NUM_STATES != 8
#error expected 8 states
#endif
#if EIGEN_SPACE_DIM != 7
#error expected 7 waves
#endif

//matches the 8-var input state 
struct cons_t {
	real rho;
	real mx;
	real my;
	real mz;
	real bx;
	real by;
	real bz;
	real ETotal;
};

struct prim_t {
	real rho;
	real4 v;
	real4 b;
	real hTotal;
};

void calcPrimFromCons(struct prim_t *W, const __global real* U_, real ePot);
void calcPrimFromCons(struct prim_t *W, const __global real* U_, real ePot) {
	const real gamma_1 = gamma - 1.;
	const __global struct cons_t* U = (const __global struct cons_t*)U_;
	real rho = U->rho; 
	real invRho = 1. / rho;
	real4 v = (real4)(U->mx, U->my, U->mz, 0.) * invRho;
	real4 b = (real4)(U->bx, U->by, U->bz, 0.);
	real ETotal = U->ETotal;
	real EMag = .5 * dot(b,b);
	real EHydro = ETotal - EMag;
	real EKin = .5 * rho * dot(v, v);
	real EPot = rho * ePot;
	real EInt = EHydro - EKin - EPot;
	real P = gamma_1 * EInt;
	real PTotal = P + EMag;
	real hTotal = (ETotal + PTotal) * invRho;
	W->rho = rho;
	W->v = v;
	W->b = b;
	W->hTotal = hTotal;
}

real2 calcRoeVars(struct prim_t *W, const struct prim_t *WL, const struct prim_t *WR);
real2 calcRoeVars(struct prim_t *W, const struct prim_t *WL, const struct prim_t *WR) {
	real sqrtRhoL = sqrt(WL->rho);
	real sqrtRhoR = sqrt(WR->rho);

	real invDenom = 1. / (sqrtRhoL + sqrtRhoR);
	W->rho = sqrtRhoL * sqrtRhoR;
	W->v = (WL->v * sqrtRhoL + WR->v * sqrtRhoR) * invDenom;
	W->b = (WL->b * sqrtRhoL + WR->b * sqrtRhoR) * invDenom;
	W->hTotal = (WL->hTotal * sqrtRhoL + WR->hTotal * sqrtRhoR) * invDenom;
	
	real4 db = WL->b - WR->b;
	real X = .5 * (db.y * db.y + db.z * db.z) * invDenom * invDenom;
	real Y = .5 * (WL->rho + WR->rho) / W->rho;

	return (real2)(X, Y);
}

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* solidBuffer,
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer,
	int side);

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* solidBuffer,
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer,
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
	int interfaceIndex = side + DIM * index;

	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + EIGEN_SPACE_DIM * interfaceIndex;
	__global real* eigenvectorsInverse = eigenvectorsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;
	__global real* eigenvectors = eigenvectorsInverse + EIGEN_SPACE_DIM * EIGEN_SPACE_DIM;

	const real gamma_1 = gamma - 1.;
	const real gamma_2 = gamma - 2.;

	struct prim_t WL, WR;
	calcPrimFromCons(&WL, stateL, potentialBuffer[indexPrev]);
	calcPrimFromCons(&WR, stateR, potentialBuffer[index]);
	
	//rotate v and magnetic field so x is forward
#if DIM > 1
	if (side == 1) {
		WL.v.xy = WL.v.yx;
		WR.b.xy = WR.b.yx;
		WL.b.xy = WL.b.yx;
		WR.b.xy = WR.b.yx;
	} 
#endif
#if DIM > 2
	else if (side == 2) {
		WL.v.xz = WL.v.zx;
		WR.b.xz = WR.b.zx;
		WL.b.xz = WL.b.zx;
		WR.b.xz = WR.b.zx;
	}
#endif

	struct prim_t W;
	real2 xy = calcRoeVars(&W, &WL, &WR);

	real rho = W.rho;
	real4 v = W.v;
	real vx = W.v.x;
	real vy = W.v.y;
	real vz = W.v.z;
	real bx = W.b.x;
	real by = W.b.y;
	real bz = W.b.z;
	real hTotal = W.hTotal;
	real X = xy.x;
	real Y = xy.y;

	real _1_rho = 1. / rho;
	real vSq = dot(v, v);
	real bPerpSq = by*by + bz*bz;
	real bStarPerpSq = (gamma_1 - gamma_2 * Y) * bPerpSq;
	real CAxSq = bx*bx*_1_rho;
	real CASq = CAxSq + bPerpSq * _1_rho;
	real hHydro = hTotal - CASq;
	real aTildeSq = max((gamma_1 * (hHydro - .5 * vSq) - gamma_2 * X), 1e-20);

	real bStarPerpSq_rho = bStarPerpSq * _1_rho;
	real CATildeSq = CAxSq + bStarPerpSq_rho;
	real CStarSq = .5 * (CATildeSq + aTildeSq);
	real CA_a_TildeSqDiff = .5 * (CATildeSq - aTildeSq);
	real sqrtDiscr = sqrt(CA_a_TildeSqDiff * CA_a_TildeSqDiff + aTildeSq * bStarPerpSq_rho);
	
	real CfSq = CStarSq + sqrtDiscr;
	real Cf = sqrt(CfSq);

	real CsSq = aTildeSq * CAxSq / CfSq;
	real Cs = sqrt(CsSq);

	real bPerpLen = sqrt(bPerpSq);
	real bStarPerpLen = sqrt(bStarPerpSq);
	real betaY, betaZ;
	if (bPerpLen == 0.) {
		betaY = 1.;
		betaZ = 0.;
	} else {
		betaY = by / bPerpLen;
		betaZ = bz / bPerpLen;
	}
	real betaStarY = betaY / sqrt(gamma_1 - gamma_2*Y);
	real betaStarZ = betaZ / sqrt(gamma_1 - gamma_2*Y);
	real betaStarSq = betaStarY*betaStarY + betaStarZ*betaStarZ;
	real vDotBeta = vy*betaStarY + vz*betaStarZ;

	real alphaF, alphaS;
	if (CfSq - CsSq == 0.) {
		alphaF = 1.;
		alphaS = 0.;
	} else if (aTildeSq - CsSq <= 0.) {
		alphaF = 0.;
		alphaS = 1.;
	} else if (CfSq - aTildeSq <= 0.) {
		alphaF = 1.;
		alphaS = 0.;
	} else {
		alphaF = sqrt((aTildeSq - CsSq) / (CfSq - CsSq));
		alphaS = sqrt((CfSq - aTildeSq) / (CfSq - CsSq));
	}

	real sqrtRho = sqrt(rho);
	real _1_sqrtRho = 1. / sqrtRho;
	real sbx = bx >= 0. ? 1. : -1.;
	real aTilde = sqrt(aTildeSq);
	real Qf = Cf*alphaF*sbx;
	real Qs = Cs*alphaS*sbx;
	real Af = aTilde*alphaF*_1_sqrtRho;
	real As = aTilde*alphaS*_1_sqrtRho;
	real Afpbb = Af*bStarPerpLen*betaStarSq;
	real Aspbb = As*bStarPerpLen*betaStarSq;

	real CAx = sqrt(CAxSq);
	eigenvalues[0] = vx - Cf;
	eigenvalues[1] = vx - CAx;
	eigenvalues[2] = vx - Cs;
	eigenvalues[3] = vx;
	eigenvalues[4] = vx + Cs;
	eigenvalues[5] = vx + CAx;
	eigenvalues[6] = vx + Cf;

	// right eigenvectors
	real qa3 = alphaF*vy;
	real qb3 = alphaS*vy;
	real qc3 = Qs*betaStarY;
	real qd3 = Qf*betaStarY;
	real qa4 = alphaF*vz;
	real qb4 = alphaS*vz;
	real qc4 = Qs*betaStarZ;
	real qd4 = Qf*betaStarZ;
	real r52 = -(vy*betaZ - vz*betaY);
	real r61 = As*betaStarY;
	real r62 = -betaZ*sbx*_1_sqrtRho;
	real r63 = -Af*betaStarY;
	real r71 = As*betaStarZ;
	real r72 = betaY*sbx*_1_sqrtRho;
	real r73 = -Af*betaStarZ;
	//rows
	eigenvectors[0 + EIGEN_SPACE_DIM * 0] = alphaF;
	eigenvectors[0 + EIGEN_SPACE_DIM * 1] = 0.;
	eigenvectors[0 + EIGEN_SPACE_DIM * 2] = alphaS;
	eigenvectors[0 + EIGEN_SPACE_DIM * 3] = 1.;
	eigenvectors[0 + EIGEN_SPACE_DIM * 4] = alphaS;
	eigenvectors[0 + EIGEN_SPACE_DIM * 5] = 0.;
	eigenvectors[0 + EIGEN_SPACE_DIM * 6] = alphaF;
	eigenvectors[1 + EIGEN_SPACE_DIM * 0] = alphaF*eigenvalues[0];
	eigenvectors[1 + EIGEN_SPACE_DIM * 1] = 0.;
	eigenvectors[1 + EIGEN_SPACE_DIM * 2] = alphaS*eigenvalues[2];
	eigenvectors[1 + EIGEN_SPACE_DIM * 3] = vx;
	eigenvectors[1 + EIGEN_SPACE_DIM * 4] = alphaS*eigenvalues[4];
	eigenvectors[1 + EIGEN_SPACE_DIM * 5] = 0.;
	eigenvectors[1 + EIGEN_SPACE_DIM * 6] = alphaF*eigenvalues[6];
	eigenvectors[2 + EIGEN_SPACE_DIM * 0] = qa3 + qc3;
	eigenvectors[2 + EIGEN_SPACE_DIM * 1] = -betaZ;
	eigenvectors[2 + EIGEN_SPACE_DIM * 2] = qb3 - qd3;
	eigenvectors[2 + EIGEN_SPACE_DIM * 3] = vy;
	eigenvectors[2 + EIGEN_SPACE_DIM * 4] = qb3 + qd3;
	eigenvectors[2 + EIGEN_SPACE_DIM * 5] = betaZ;
	eigenvectors[2 + EIGEN_SPACE_DIM * 6] = qa3 - qc3;
	eigenvectors[3 + EIGEN_SPACE_DIM * 0] = qa4 + qc4;
	eigenvectors[3 + EIGEN_SPACE_DIM * 1] = betaY;
	eigenvectors[3 + EIGEN_SPACE_DIM * 2] = qb4 - qd4;
	eigenvectors[3 + EIGEN_SPACE_DIM * 3] = vz;
	eigenvectors[3 + EIGEN_SPACE_DIM * 4] = qb4 + qd4;
	eigenvectors[3 + EIGEN_SPACE_DIM * 5] = -betaY;
	eigenvectors[3 + EIGEN_SPACE_DIM * 6] = qa4 - qc4;
	eigenvectors[4 + EIGEN_SPACE_DIM * 0] = alphaF*(hHydro - vx*Cf) + Qs*vDotBeta + Aspbb;
	eigenvectors[4 + EIGEN_SPACE_DIM * 1] = r52;
	eigenvectors[4 + EIGEN_SPACE_DIM * 2] = alphaS*(hHydro - vx*Cs) - Qf*vDotBeta - Afpbb;
	eigenvectors[4 + EIGEN_SPACE_DIM * 3] = .5*vSq + gamma_2*X/gamma_1;
	eigenvectors[4 + EIGEN_SPACE_DIM * 4] = alphaS*(hHydro + vx*Cs) + Qf*vDotBeta - Afpbb;
	eigenvectors[4 + EIGEN_SPACE_DIM * 5] = -r52;
	eigenvectors[4 + EIGEN_SPACE_DIM * 6] = alphaF*(hHydro + vx*Cf) - Qs*vDotBeta + Aspbb;
	eigenvectors[5 + EIGEN_SPACE_DIM * 0] = r61;
	eigenvectors[5 + EIGEN_SPACE_DIM * 1] = r62;
	eigenvectors[5 + EIGEN_SPACE_DIM * 2] = r63;
	eigenvectors[5 + EIGEN_SPACE_DIM * 3] = 0.;
	eigenvectors[5 + EIGEN_SPACE_DIM * 4] = r63;
	eigenvectors[5 + EIGEN_SPACE_DIM * 5] = r62;
	eigenvectors[5 + EIGEN_SPACE_DIM * 6] = r61;
	eigenvectors[6 + EIGEN_SPACE_DIM * 0] = r71;
	eigenvectors[6 + EIGEN_SPACE_DIM * 1] = r72;
	eigenvectors[6 + EIGEN_SPACE_DIM * 2] = r73;
	eigenvectors[6 + EIGEN_SPACE_DIM * 3] = 0.;
	eigenvectors[6 + EIGEN_SPACE_DIM * 4] = r73;
	eigenvectors[6 + EIGEN_SPACE_DIM * 5] = r72;
	eigenvectors[6 + EIGEN_SPACE_DIM * 6] = r71;

	// left eigenvectors
	real norm = .5/aTildeSq;
	real Cff = norm*alphaF*Cf;
	real Css = norm*alphaS*Cs;
	Qf = Qf * norm;
	Qs = Qs * norm;
	real AHatF = norm*Af*rho;
	real AHatS = norm*As*rho;
	real afpb = norm*Af*bStarPerpLen;
	real aspb = norm*As*bStarPerpLen;

	norm = norm * gamma_1;
	alphaF = alphaF * norm;
	alphaS = alphaS * norm;
	real QStarY = betaStarY/betaStarSq;
	real QStarZ = betaStarZ/betaStarSq;
	real vqstr = (vy*QStarY + vz*QStarZ);
	norm = norm * 2.;
	//rows
	eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 0] = alphaF*(vSq-hHydro) + Cff*(Cf+vx) - Qs*vqstr - aspb;
	eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 1] = -alphaF*vx - Cff;
	eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 2] = -alphaF*vy + Qs*QStarY;
	eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 3] = -alphaF*vz + Qs*QStarZ;
	eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 4] = alphaF;
	eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 5] = AHatS*QStarY - alphaF*by;
	eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 6] = AHatS*QStarZ - alphaF*bz;
	eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 0] = .5*(vy*betaZ - vz*betaY);
	eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 1] = 0.;
	eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 2] = -.5*betaZ;
	eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 3] = .5*betaY;
	eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 4] = 0.;
	eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 5] = -.5*sqrtRho*betaZ*sbx;
	eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 6] = .5*sqrtRho*betaY*sbx;
	eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 0] = alphaS*(vSq-hHydro) + Css*(Cs+vx) + Qf*vqstr + afpb;
	eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 1] = -alphaS*vx - Css;
	eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 2] = -alphaS*vy - Qf*QStarY;
	eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 3] = -alphaS*vz - Qf*QStarZ;
	eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 4] = alphaS;
	eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 5] = -AHatF*QStarY - alphaS*by;
	eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 6] = -AHatF*QStarZ - alphaS*bz;
	eigenvectorsInverse[3 + EIGEN_SPACE_DIM * 0] = 1. - norm*(.5*vSq - gamma_2*X/gamma_1) ;
	eigenvectorsInverse[3 + EIGEN_SPACE_DIM * 1] = norm*vx;
	eigenvectorsInverse[3 + EIGEN_SPACE_DIM * 2] = norm*vy;
	eigenvectorsInverse[3 + EIGEN_SPACE_DIM * 3] = norm*vz;
	eigenvectorsInverse[3 + EIGEN_SPACE_DIM * 4] = -norm;
	eigenvectorsInverse[3 + EIGEN_SPACE_DIM * 5] = norm*by;
	eigenvectorsInverse[3 + EIGEN_SPACE_DIM * 6] = norm*bz;
	eigenvectorsInverse[4 + EIGEN_SPACE_DIM * 0] = alphaS*(vSq-hHydro) + Css*(Cs-vx) - Qf*vqstr + afpb;
	eigenvectorsInverse[4 + EIGEN_SPACE_DIM * 1] = -alphaS*vx + Css;
	eigenvectorsInverse[4 + EIGEN_SPACE_DIM * 2] = -alphaS*vy + Qf*QStarY;
	eigenvectorsInverse[4 + EIGEN_SPACE_DIM * 3] = -alphaS*vz + Qf*QStarZ;
	eigenvectorsInverse[4 + EIGEN_SPACE_DIM * 4] = alphaS;
	eigenvectorsInverse[4 + EIGEN_SPACE_DIM * 5] = eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 5];
	eigenvectorsInverse[4 + EIGEN_SPACE_DIM * 6] = eigenvectorsInverse[2 + EIGEN_SPACE_DIM * 6];
	eigenvectorsInverse[5 + EIGEN_SPACE_DIM * 0] = -eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 0];
	eigenvectorsInverse[5 + EIGEN_SPACE_DIM * 1] = 0.;
	eigenvectorsInverse[5 + EIGEN_SPACE_DIM * 2] = -eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 2];
	eigenvectorsInverse[5 + EIGEN_SPACE_DIM * 3] = -eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 3];
	eigenvectorsInverse[5 + EIGEN_SPACE_DIM * 4] = 0.;
	eigenvectorsInverse[5 + EIGEN_SPACE_DIM * 5] = eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 5];
	eigenvectorsInverse[5 + EIGEN_SPACE_DIM * 6] = eigenvectorsInverse[1 + EIGEN_SPACE_DIM * 6];
	eigenvectorsInverse[6 + EIGEN_SPACE_DIM * 0] = alphaF*(vSq-hHydro) + Cff*(Cf-vx) + Qs*vqstr - aspb;
	eigenvectorsInverse[6 + EIGEN_SPACE_DIM * 1] = -alphaF*vx + Cff;
	eigenvectorsInverse[6 + EIGEN_SPACE_DIM * 2] = -alphaF*vy - Qs*QStarY;
	eigenvectorsInverse[6 + EIGEN_SPACE_DIM * 3] = -alphaF*vz - Qs*QStarZ;
	eigenvectorsInverse[6 + EIGEN_SPACE_DIM * 4] = alphaF;
	eigenvectorsInverse[6 + EIGEN_SPACE_DIM * 5] = eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 5];
	eigenvectorsInverse[6 + EIGEN_SPACE_DIM * 6] = eigenvectorsInverse[0 + EIGEN_SPACE_DIM * 6];

#if 0	//instead I'm going to rotate upon application of eigenvectors
#if DIM > 1
	if (side == 1) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;

			//each row's xy <- yx 
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y] = tmp;
			
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Y] = tmp;
			
			//each column's xy <- yx
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i] = tmp;
			
			tmp = eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i] = eigenvectors[STATE_MAGNETIC_FIELD_Y + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z] = tmp;
			
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Z] = tmp;
			
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i] = tmp;
			
			tmp = eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i] = eigenvectors[STATE_MAGNETIC_FIELD_Z + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_Z + NUM_STATES * i] = tmp;
		}
	}
#endif
#endif
#endif
}

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* solidBuffer,
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer)
{
	for (int side = 0; side < DIM; ++side) {
		calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, stateBuffer, potentialBuffer, solidBuffer, fluxBuffer, fluxFlagBuffer, side);
	}
}

void leftEigenvectorTransform(
	real* results_,
	const __global real* eigenvectorsBuffer,
	const real* input_,
	int side);

void leftEigenvectorTransform(
	real* results,
	const __global real* eigenvectorsBuffer,
	const real* input_,
	int side)
{
	real tmp;
	const __global real* eigenvectorsInverse = eigenvectorsBuffer;

	real input[8] = {
		input_[0],
		input_[1],
		input_[2],
		input_[3],
		input_[4],
		input_[5],
		input_[6],
		input_[7],
	};
	//swap x and side
//	tmp = input[1]; input[1] = input[1+side]; input[1+side] = tmp;
//	tmp = input[4]; input[4] = input[4+side]; input[4+side] = tmp;
	//set 8th to 5th (and ignore 8th)
	input[4] = input[7];

	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		real sum = 0.;
		for (int j = 0; j < EIGEN_SPACE_DIM; ++j) {
			sum += eigenvectorsInverse[i + EIGEN_SPACE_DIM * j] * input[j];
		}
		results[i] = sum;
	}

	//set 8th to 5th, set 5th to 0
	results[7] = results[4];
	results[4] = 0;
	//swap x and side
//	tmp = results[1]; results[1] = results[1+side]; results[1+side] = tmp;
//	tmp = results[4]; results[4] = results[4+side]; results[4+side] = tmp;
}

void rightEigenvectorTransform(
	__global real* results,
	const __global real* eigenvectorsBuffer,
	const real* input_,
	int side);

void rightEigenvectorTransform(
	__global real* results,
	const __global real* eigenvectorsBuffer,
	const real* input_,
	int side)
{
	real tmp;
	const __global real* eigenvectors = eigenvectorsBuffer + EIGEN_SPACE_DIM * EIGEN_SPACE_DIM;

	real input[8] = {
		input_[0],
		input_[1],
		input_[2],
		input_[3],
		input_[4],
		input_[5],
		input_[6],
		input_[7],
	};
	//swap x and side
//	tmp = input[1]; input[1] = input[1+side]; input[1+side] = tmp;
//	tmp = input[4]; input[4] = input[4+side]; input[4+side] = tmp;
	//set 8th to 5th (and ignore 8th)
	input[4] = input[7];

	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		real sum = 0.;
		for (int j = 0; j < EIGEN_SPACE_DIM; ++j) {
			sum += eigenvectors[i + EIGEN_SPACE_DIM * j] * input[j];
		}
		results[i] = sum;
	}

	//set 8th to 5th, set 5th to 0
	results[7] = results[4];
	results[4] = 0;
	//swap x and side
//	tmp = results[1]; results[1] = results[1+side]; results[1+side] = tmp;
//	tmp = results[4]; results[4] = results[4+side]; results[4+side] = tmp;
}

//just like calcFlux except if the flux flag is already set then don't do it
__kernel void calcMHDFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenvectorsBuffer,
	const __global real* deltaQTildeBuffer,
	real dt,
#ifdef SOLID
	const __global char* solidBuffer,
#endif	//SOLID
	const __global char* fluxFlagBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
//	if (fluxFlagBuffer[side + DIM * index]) return;
	
	calcFlux(
		fluxBuffer,
		stateBuffer,
		eigenvaluesBuffer,
		eigenvectorsBuffer,
		deltaQTildeBuffer,
		dt
#ifdef SOLID
		, solidBuffer
#endif	//SOLID
	);
}

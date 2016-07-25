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
#if EIGEN_TRANSFORM_STRUCT_SIZE != 7*7*2
#error expected EIGEN_TRANSFORM_STRUCT_SIZE to be 2 dense 7x7 matrices
#endif

//matches the 8-var input state 
struct cons_t {
	real rho;
	real mx, my, mz;
	real bx, by, bz;
	real ETotal;
};

struct Roe_t {
	real rho;
	real vx, vy, vz;
	real bx, by, bz;
	real hTotal;
	real X;
	real Y;
};

void calcPrimFromCons(struct Roe_t *W, const __global struct cons_t *U, real ePot);
void calcPrimFromCons(struct Roe_t *W, const __global struct cons_t *U, real ePot) {
	W->rho = U->rho;
	W->vx = U->mx / U->rho;
	W->vy = U->my / U->rho;
	W->vz = U->mz / U->rho;
	W->bx = U->bx;
	W->by = U->by;
	W->bz = U->bz;
	real vSq = W->vx * W->vx + W->vy * W->vy + W->vz * W->vz;
	real bSq = W->bx * W->bx + W->by * W->by + W->bz * W->bz;
	real EKin = .5 * U->rho * vSq;
	real EMag = .5 * bSq;
	real P = (U->ETotal - EKin - EMag) * (gamma - 1.);
	W->rho = max(W->rho, 1e-7);
	P = max(P, 1e-7);
	W->hTotal = (U->ETotal + P + EMag) / U->rho;
}

void calcRoeAverage(struct Roe_t *W, const struct Roe_t *WL, const struct Roe_t *WR);
void calcRoeAverage(struct Roe_t *W, const struct Roe_t *WL, const struct Roe_t *WR) {
	real sqrtRhoL = sqrt(WL->rho);
	real sqrtRhoR = sqrt(WR->rho);

	real invDenom = 1. / (sqrtRhoL + sqrtRhoR);
	W->rho = sqrtRhoL * sqrtRhoR;
	W->vx = (WL->vx * sqrtRhoL + WR->vx * sqrtRhoR) * invDenom;
	W->vy = (WL->vy * sqrtRhoL + WR->vy * sqrtRhoR) * invDenom;
	W->vz = (WL->vz * sqrtRhoL + WR->vz * sqrtRhoR) * invDenom;
	W->bx = (WL->bx * sqrtRhoL + WR->bx * sqrtRhoR) * invDenom;

	//by and bz weights are switched
	W->by = (WL->by * sqrtRhoR + WR->by * sqrtRhoL) * invDenom;
	W->bz = (WL->bz * sqrtRhoR + WR->bz * sqrtRhoL) * invDenom;
	
	W->hTotal = (WL->hTotal * sqrtRhoL + WR->hTotal * sqrtRhoR) * invDenom;
	real dby = WL->by - WR->by;
	real dbz = WL->bz - WR->bz;
	W->X = .5 * (dby * dby + dbz * dbz) * invDenom * invDenom;
	W->Y = .5 * (WL->rho + WR->rho) / W->rho;
}

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
#ifdef SOLID
	const __global char* solidBuffer,
#endif
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer,
	int side);

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
#ifdef SOLID
	const __global char* solidBuffer,
#endif
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0.);
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
	const __global struct cons_t *UL = (const __global struct cons_t*)(stateBuffer + NUM_STATES * indexPrev);
	const __global struct cons_t *UR = (const __global struct cons_t*)(stateBuffer + NUM_STATES * index);
	
	int interfaceIndex = side + DIM * index;
	__global real* eigenvalues = eigenvaluesBuffer + EIGEN_SPACE_DIM * interfaceIndex;
	__global real* leftEigenvectors = eigenvectorsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;
	__global real* rightEigenvectors = leftEigenvectors + EIGEN_SPACE_DIM * EIGEN_SPACE_DIM;

	const real gamma_1 = gamma - 1.;
	const real gamma_2 = gamma - 2.;

	struct Roe_t WL, WR;
	calcPrimFromCons(&WL, UL, potentialBuffer[indexPrev]);
	calcPrimFromCons(&WR, UR, potentialBuffer[index]);

/*
	//rotate v and magnetic field so x is forward
#if DIM > 1
	if (side == 1) {
		real tmp;
		tmp = WL.vx; WL.vx = WL.vy; WL.vy = tmp;
		tmp = WR.vx; WR.vx = WR.vy; WR.vy = tmp;
		tmp = WL.bx; WL.bx = WL.by; WL.by = tmp;
		tmp = WR.bx; WR.bx = WR.by; WR.by = tmp;
	} 
#endif
#if DIM > 2
	else if (side == 2) {
		real tmp;
		tmp = WL.vx; WL.vx = WL.vz; WL.vz = tmp;
		tmp = WR.vx; WR.vx = WR.vz; WR.vz = tmp;
		tmp = WL.bx; WL.bx = WL.bz; WL.bz = tmp;
		tmp = WR.bx; WR.bx = WR.bz; WR.bz = tmp;
	}
#endif
*/

	struct Roe_t W;
	calcRoeAverage(&W, &WL, &WR);

	real rho = W.rho;
	real vx = W.vx;
	real vy = W.vy;
	real vz = W.vz;
	real vSq = vx * vx + vy * vy + vz * vz;
	real bx = W.bx;
	real by = W.by;
	real bz = W.bz;
	real hTotal = W.hTotal;
	real X = W.X;
	real Y = W.Y;

	real _1_rho = 1. / rho;
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
	real lambdaFastMin = vx - Cf;
	real lambdaSlowMin = vx - Cs;
	real lambdaSlowMax = vx + Cs;
	real lambdaFastMax = vx + Cf;
	eigenvalues[0] = lambdaFastMin;
	eigenvalues[1] = vx - CAx;
	eigenvalues[2] = lambdaSlowMin;
	eigenvalues[3] = vx;
	eigenvalues[4] = lambdaSlowMax;
	eigenvalues[5] = vx + CAx;
	eigenvalues[6] = lambdaFastMax;

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
	rightEigenvectors[0 + EIGEN_SPACE_DIM * 0] = alphaF;
	rightEigenvectors[0 + EIGEN_SPACE_DIM * 1] = 0.;
	rightEigenvectors[0 + EIGEN_SPACE_DIM * 2] = alphaS;
	rightEigenvectors[0 + EIGEN_SPACE_DIM * 3] = 1.;
	rightEigenvectors[0 + EIGEN_SPACE_DIM * 4] = alphaS;
	rightEigenvectors[0 + EIGEN_SPACE_DIM * 5] = 0.;
	rightEigenvectors[0 + EIGEN_SPACE_DIM * 6] = alphaF;
	rightEigenvectors[1 + EIGEN_SPACE_DIM * 0] = alphaF*lambdaFastMin;
	rightEigenvectors[1 + EIGEN_SPACE_DIM * 1] = 0.;
	rightEigenvectors[1 + EIGEN_SPACE_DIM * 2] = alphaS*lambdaSlowMin;
	rightEigenvectors[1 + EIGEN_SPACE_DIM * 3] = vx;
	rightEigenvectors[1 + EIGEN_SPACE_DIM * 4] = alphaS*lambdaSlowMax;
	rightEigenvectors[1 + EIGEN_SPACE_DIM * 5] = 0.;
	rightEigenvectors[1 + EIGEN_SPACE_DIM * 6] = alphaF*lambdaFastMax;
	rightEigenvectors[2 + EIGEN_SPACE_DIM * 0] = qa3 + qc3;
	rightEigenvectors[2 + EIGEN_SPACE_DIM * 1] = -betaZ;
	rightEigenvectors[2 + EIGEN_SPACE_DIM * 2] = qb3 - qd3;
	rightEigenvectors[2 + EIGEN_SPACE_DIM * 3] = vy;
	rightEigenvectors[2 + EIGEN_SPACE_DIM * 4] = qb3 + qd3;
	rightEigenvectors[2 + EIGEN_SPACE_DIM * 5] = betaZ;
	rightEigenvectors[2 + EIGEN_SPACE_DIM * 6] = qa3 - qc3;
	rightEigenvectors[3 + EIGEN_SPACE_DIM * 0] = qa4 + qc4;
	rightEigenvectors[3 + EIGEN_SPACE_DIM * 1] = betaY;
	rightEigenvectors[3 + EIGEN_SPACE_DIM * 2] = qb4 - qd4;
	rightEigenvectors[3 + EIGEN_SPACE_DIM * 3] = vz;
	rightEigenvectors[3 + EIGEN_SPACE_DIM * 4] = qb4 + qd4;
	rightEigenvectors[3 + EIGEN_SPACE_DIM * 5] = -betaY;
	rightEigenvectors[3 + EIGEN_SPACE_DIM * 6] = qa4 - qc4;
	rightEigenvectors[4 + EIGEN_SPACE_DIM * 0] = alphaF*(hHydro - vx*Cf) + Qs*vDotBeta + Aspbb;
	rightEigenvectors[4 + EIGEN_SPACE_DIM * 1] = r52;
	rightEigenvectors[4 + EIGEN_SPACE_DIM * 2] = alphaS*(hHydro - vx*Cs) - Qf*vDotBeta - Afpbb;
	rightEigenvectors[4 + EIGEN_SPACE_DIM * 3] = .5*vSq + gamma_2*X/gamma_1;
	rightEigenvectors[4 + EIGEN_SPACE_DIM * 4] = alphaS*(hHydro + vx*Cs) + Qf*vDotBeta - Afpbb;
	rightEigenvectors[4 + EIGEN_SPACE_DIM * 5] = -r52;
	rightEigenvectors[4 + EIGEN_SPACE_DIM * 6] = alphaF*(hHydro + vx*Cf) - Qs*vDotBeta + Aspbb;
	rightEigenvectors[5 + EIGEN_SPACE_DIM * 0] = r61;
	rightEigenvectors[5 + EIGEN_SPACE_DIM * 1] = r62;
	rightEigenvectors[5 + EIGEN_SPACE_DIM * 2] = r63;
	rightEigenvectors[5 + EIGEN_SPACE_DIM * 3] = 0.;
	rightEigenvectors[5 + EIGEN_SPACE_DIM * 4] = r63;
	rightEigenvectors[5 + EIGEN_SPACE_DIM * 5] = r62;
	rightEigenvectors[5 + EIGEN_SPACE_DIM * 6] = r61;
	rightEigenvectors[6 + EIGEN_SPACE_DIM * 0] = r71;
	rightEigenvectors[6 + EIGEN_SPACE_DIM * 1] = r72;
	rightEigenvectors[6 + EIGEN_SPACE_DIM * 2] = r73;
	rightEigenvectors[6 + EIGEN_SPACE_DIM * 3] = 0.;
	rightEigenvectors[6 + EIGEN_SPACE_DIM * 4] = r73;
	rightEigenvectors[6 + EIGEN_SPACE_DIM * 5] = r72;
	rightEigenvectors[6 + EIGEN_SPACE_DIM * 6] = r71;

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
	
	real l16 = AHatS*QStarY - alphaF*by;
	real l17 = AHatS*QStarZ - alphaF*bz;
	real l21 = .5*(vy*betaZ - vz*betaY);
	real l23 = .5*betaZ;
	real l24 = .5*betaY;
	real l26 = -.5*sqrtRho*betaZ*sbx;
	real l27 = .5*sqrtRho*betaY*sbx;
	real l36 = -AHatF*QStarY - alphaS*by;
	real l37 = -AHatF*QStarZ - alphaS*bz;
	//rows
	leftEigenvectors[0 + EIGEN_SPACE_DIM * 0] = alphaF*(vSq-hHydro) + Cff*(Cf+vx) - Qs*vqstr - aspb;
	leftEigenvectors[0 + EIGEN_SPACE_DIM * 1] = -alphaF*vx - Cff;
	leftEigenvectors[0 + EIGEN_SPACE_DIM * 2] = -alphaF*vy + Qs*QStarY;
	leftEigenvectors[0 + EIGEN_SPACE_DIM * 3] = -alphaF*vz + Qs*QStarZ;
	leftEigenvectors[0 + EIGEN_SPACE_DIM * 4] = alphaF;
	leftEigenvectors[0 + EIGEN_SPACE_DIM * 5] = l16;
	leftEigenvectors[0 + EIGEN_SPACE_DIM * 6] = l17;
	leftEigenvectors[1 + EIGEN_SPACE_DIM * 0] = l21;
	leftEigenvectors[1 + EIGEN_SPACE_DIM * 1] = 0.;
	leftEigenvectors[1 + EIGEN_SPACE_DIM * 2] = -l23;
	leftEigenvectors[1 + EIGEN_SPACE_DIM * 3] = l24;
	leftEigenvectors[1 + EIGEN_SPACE_DIM * 4] = 0.;
	leftEigenvectors[1 + EIGEN_SPACE_DIM * 5] = l26;
	leftEigenvectors[1 + EIGEN_SPACE_DIM * 6] = l27;
	leftEigenvectors[2 + EIGEN_SPACE_DIM * 0] = alphaS*(vSq-hHydro) + Css*(Cs+vx) + Qf*vqstr + afpb;
	leftEigenvectors[2 + EIGEN_SPACE_DIM * 1] = -alphaS*vx - Css;
	leftEigenvectors[2 + EIGEN_SPACE_DIM * 2] = -alphaS*vy - Qf*QStarY;
	leftEigenvectors[2 + EIGEN_SPACE_DIM * 3] = -alphaS*vz - Qf*QStarZ;
	leftEigenvectors[2 + EIGEN_SPACE_DIM * 4] = alphaS;
	leftEigenvectors[2 + EIGEN_SPACE_DIM * 5] = l36;
	leftEigenvectors[2 + EIGEN_SPACE_DIM * 6] = l37;
	leftEigenvectors[3 + EIGEN_SPACE_DIM * 0] = 1. - norm*(.5*vSq - gamma_2*X/gamma_1) ;
	leftEigenvectors[3 + EIGEN_SPACE_DIM * 1] = norm*vx;
	leftEigenvectors[3 + EIGEN_SPACE_DIM * 2] = norm*vy;
	leftEigenvectors[3 + EIGEN_SPACE_DIM * 3] = norm*vz;
	leftEigenvectors[3 + EIGEN_SPACE_DIM * 4] = -norm;
	leftEigenvectors[3 + EIGEN_SPACE_DIM * 5] = norm*by;
	leftEigenvectors[3 + EIGEN_SPACE_DIM * 6] = norm*bz;
	leftEigenvectors[4 + EIGEN_SPACE_DIM * 0] = alphaS*(vSq-hHydro) + Css*(Cs-vx) - Qf*vqstr + afpb;
	leftEigenvectors[4 + EIGEN_SPACE_DIM * 1] = -alphaS*vx + Css;
	leftEigenvectors[4 + EIGEN_SPACE_DIM * 2] = -alphaS*vy + Qf*QStarY;
	leftEigenvectors[4 + EIGEN_SPACE_DIM * 3] = -alphaS*vz + Qf*QStarZ;
	leftEigenvectors[4 + EIGEN_SPACE_DIM * 4] = alphaS;
	leftEigenvectors[4 + EIGEN_SPACE_DIM * 5] = l36;
	leftEigenvectors[4 + EIGEN_SPACE_DIM * 6] = l37;
	leftEigenvectors[5 + EIGEN_SPACE_DIM * 0] = -l21;
	leftEigenvectors[5 + EIGEN_SPACE_DIM * 1] = 0.;
	leftEigenvectors[5 + EIGEN_SPACE_DIM * 2] = -l23;
	leftEigenvectors[5 + EIGEN_SPACE_DIM * 3] = -l24;
	leftEigenvectors[5 + EIGEN_SPACE_DIM * 4] = 0.;
	leftEigenvectors[5 + EIGEN_SPACE_DIM * 5] = l26;
	leftEigenvectors[5 + EIGEN_SPACE_DIM * 6] = l27;
	leftEigenvectors[6 + EIGEN_SPACE_DIM * 0] = alphaF*(vSq-hHydro) + Cff*(Cf-vx) + Qs*vqstr - aspb;
	leftEigenvectors[6 + EIGEN_SPACE_DIM * 1] = -alphaF*vx + Cff;
	leftEigenvectors[6 + EIGEN_SPACE_DIM * 2] = -alphaF*vy - Qs*QStarY;
	leftEigenvectors[6 + EIGEN_SPACE_DIM * 3] = -alphaF*vz - Qs*QStarZ;
	leftEigenvectors[6 + EIGEN_SPACE_DIM * 4] = alphaF;
	leftEigenvectors[6 + EIGEN_SPACE_DIM * 5] = l16;
	leftEigenvectors[6 + EIGEN_SPACE_DIM * 6] = l17;

#if 0	//instead I'm going to rotate upon application of eigenvectors
#if DIM > 1
	if (side == 1) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;

			//each row's xy <- yx 
			tmp = leftEigenvectors[i + NUM_STATES * STATE_MOMENTUM_X];
			leftEigenvectors[i + NUM_STATES * STATE_MOMENTUM_X] = leftEigenvectors[i + NUM_STATES * STATE_MOMENTUM_Y];
			leftEigenvectors[i + NUM_STATES * STATE_MOMENTUM_Y] = tmp;
			
			tmp = leftEigenvectors[i + NUM_STATES * STATE_MAGNETIC_FIELD_X];
			leftEigenvectors[i + NUM_STATES * STATE_MAGNETIC_FIELD_X] = leftEigenvectors[i + NUM_STATES * STATE_MAGNETIC_FIELD_Y];
			leftEigenvectors[i + NUM_STATES * STATE_MAGNETIC_FIELD_Y] = tmp;
			
			//each column's xy <- yx
			tmp = rightEigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			rightEigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = rightEigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i];
			rightEigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i] = tmp;
			
			tmp = rightEigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i];
			rightEigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i] = rightEigenvectors[STATE_MAGNETIC_FIELD_Y + NUM_STATES * i];
			rightEigenvectors[STATE_MAGNETIC_FIELD_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			
			tmp = leftEigenvectors[i + NUM_STATES * STATE_MOMENTUM_X];
			leftEigenvectors[i + NUM_STATES * STATE_MOMENTUM_X] = leftEigenvectors[i + NUM_STATES * STATE_MOMENTUM_Z];
			leftEigenvectors[i + NUM_STATES * STATE_MOMENTUM_Z] = tmp;
			
			tmp = leftEigenvectors[i + NUM_STATES * STATE_MAGNETIC_FIELD_X];
			leftEigenvectors[i + NUM_STATES * STATE_MAGNETIC_FIELD_X] = leftEigenvectors[i + NUM_STATES * STATE_MAGNETIC_FIELD_Z];
			leftEigenvectors[i + NUM_STATES * STATE_MAGNETIC_FIELD_Z] = tmp;
			
			tmp = rightEigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			rightEigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = rightEigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i];
			rightEigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i] = tmp;
			
			tmp = rightEigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i];
			rightEigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i] = rightEigenvectors[STATE_MAGNETIC_FIELD_Z + NUM_STATES * i];
			rightEigenvectors[STATE_MAGNETIC_FIELD_Z + NUM_STATES * i] = tmp;
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
#ifdef SOLID
	const __global char* solidBuffer,
#endif
	__global real* fluxBuffer,
	__global char* fluxFlagBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0.);
	int index = INDEXV(i);
	const __global real *U = stateBuffer + NUM_STATES * index;
	
	for (int side = 0; side < DIM; ++side) {
		calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, stateBuffer, potentialBuffer, 
#ifdef SOLID
			solidBuffer,
#endif
			fluxBuffer, fluxFlagBuffer, side);
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
	const __global real* leftEigenvectors = eigenvectorsBuffer;

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
	real tmp;
	tmp = input[1]; input[1] = input[1+side]; input[1+side] = tmp;
	tmp = input[4]; input[4] = input[4+side]; input[4+side] = tmp;
	//set 8th to 5th (and ignore 8th)
	input[4] = input[7];

	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		real sum = 0.;
		for (int j = 0; j < EIGEN_SPACE_DIM; ++j) {
			sum += leftEigenvectors[i + EIGEN_SPACE_DIM * j] * input[j];
		}
		results[i] = sum;
	}
}

void rightEigenvectorTransform(
	__global real* results,
	const __global real* eigenvectorsBuffer,
	const real* input_,
	int side);

void rightEigenvectorTransform(
	__global real* results,
	const __global real* eigenvectorsBuffer,
	const real* input,
	int side)
{
	const __global real* rightEigenvectors = eigenvectorsBuffer + EIGEN_SPACE_DIM * EIGEN_SPACE_DIM;

	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		real sum = 0.;
		for (int j = 0; j < EIGEN_SPACE_DIM; ++j) {
			sum += rightEigenvectors[i + EIGEN_SPACE_DIM * j] * input[j];
		}
		results[i] = sum;
	}

	//set 8th to 5th, set 5th to 0
	results[7] = results[4];
	results[4] = 0;
	//swap x and side
	real tmp;
	tmp = results[1]; results[1] = results[1+side]; results[1+side] = tmp;
	tmp = results[4]; results[4] = results[4+side]; results[4+side] = tmp;
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
//if I'm going to override unphysical states with the HLLC solver:
//	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
//	int index = INDEXV(i);
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

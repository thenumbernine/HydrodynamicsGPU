/*
using the following:
"Eigenvalues, Eigenvectors, and Symmetrization of the Magneto-Hydrodynamic (MHD) Equations" by Jameson 2006

Sticking with this because I like the thought of using the right eigenvectors for the left eigenvectors
(courtesy of the symmetrized transformation in the Jameson paper)

Following it in my Maxima worksheet to verify results.
https://github.com/thenumbernine/MaximaWorksheets

have dug through
"A Numerical Solution of Hyperbolic Partial Differential Equations", Trangenstein, 2007
"A multidimensional upwind scheme for magnetohydrodynamics" by Falle, Komissarov, Joarder, 1998
"A Solution-Adaptive Upwind Scheme for Ideal Magnetohydrodynamics" by Powell, Roe, Linde, Gombosi, Zeeuw, 1999
*/

#include "HydroGPU/Shared/Common.h"

#define M_SQRT_1_2	0.7071067811865475727373109293694142252206802368164f
#define M_SQRT_2	(2.f * M_SQRT_1_2)

//debugging
#define DEBUG_OUTPUT
#define DEBUG_INDEX		512

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	int side);

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	
	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;

	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
	__global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
	
	const real gammaMinusOne = gamma - 1.f;

	real densityL = stateL[STATE_DENSITY];
	real4 velocityL = VELOCITY(stateL);
	real4 magneticFieldL = (real4)(stateL[STATE_MAGNETIC_FIELD_X], stateL[STATE_MAGNETIC_FIELD_Y], stateL[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticEnergyDensityL = .5f * dot(magneticFieldL, magneticFieldL) / vaccuumPermeability;
	real totalPlasmaEnergyDensityL = stateL[STATE_ENERGY_TOTAL];
	real totalHydroEnergyDensityL = totalPlasmaEnergyDensityL - magneticEnergyDensityL;
	real kineticEnergyDensityL = .5f * densityL * dot(velocityL, velocityL);
	real potentialEnergyL = potentialBuffer[indexPrev];
	real potentialEnergyDensityL = densityL * potentialEnergyL; 
	real internalEnergyDensityL = totalHydroEnergyDensityL - kineticEnergyDensityL - potentialEnergyDensityL;
	real pressureL = gammaMinusOne * internalEnergyDensityL;
	real roeWeightL = .5f;//sqrt(densityL);

	real densityR = stateR[STATE_DENSITY];
	real4 velocityR = VELOCITY(stateR);
	real4 magneticFieldR = (real4)(stateR[STATE_MAGNETIC_FIELD_X], stateR[STATE_MAGNETIC_FIELD_Y], stateR[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticEnergyDensityR = .5f * dot(magneticFieldR, magneticFieldR) / vaccuumPermeability;
	real totalPlasmaEnergyDensityR = stateR[STATE_ENERGY_TOTAL];
	real totalHydroEnergyDensityR = totalPlasmaEnergyDensityR - magneticEnergyDensityR;
	real kineticEnergyDensityR = .5f * densityR * dot(velocityR, velocityR);
	real potentialEnergyR = potentialBuffer[index];
	real potentialEnergyDensityR = densityR * potentialEnergyR;
	real internalEnergyDensityR = totalHydroEnergyDensityR - kineticEnergyDensityR - potentialEnergyDensityR;
	real pressureR = gammaMinusOne * internalEnergyDensityR;
	real roeWeightR = .5f;//sqrt(densityR);

	//3.5.2 "In this paper, a simple arithmetic averaging of the primitive variables is done to compute the interface state."
	//but while I'm solving the degeneracy case, I'll use Roe weighting
	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real4 velocity = (velocityL * roeWeightL + velocityR * roeWeightR) * roeWeightNormalization;
	real4 magneticField = (magneticFieldL * roeWeightL + magneticFieldR * roeWeightR) * roeWeightNormalization;
	real pressure = (pressureL * roeWeightL + pressureR * roeWeightR) * roeWeightNormalization;
	real density = roeWeightL * roeWeightR;	//specific to Euler Roe weighting
	
#if DIM > 1
	if (side == 1) {
		// -90' rotation to put the y axis contents into the x axis
		velocity = (real4)(velocity.y, -velocity.x, velocity.z, 0.f);
		magneticField = (real4)(magneticField.y, -magneticField.x, magneticField.z, 0.f);
	} 
#if DIM > 2
	else if (side == 2) {
		//-90' rotation to put the z axis in the x axis
		velocity = (real4)(velocity.z, velocity.y, -velocity.x, 0.f);
		magneticField = (real4)(magneticField.z, magneticField.y, -magneticField.x, 0.f);
	}
#endif
#endif

	real velocitySq = dot(velocity, velocity);

	real4 magneticFieldT = (real4)(0.f, magneticField.y, magneticField.z, 0.f);
	real magneticFieldXSq = magneticField.x * magneticField.x;
	real magneticFieldTSq = magneticField.y * magneticField.y + magneticField.z * magneticField.z;
	real magneticFieldSq = magneticFieldXSq + magneticFieldTSq;
	real magneticFieldTLen = sqrt(magneticFieldTSq);
	
	real sqrtDensity = sqrt(density);
	
	//matrices are stored as A_ij = A[i + height * j]

	real speedOfSound = sqrt(gamma * pressure / density);
	real speedOfSoundSq = speedOfSound * speedOfSound;
	
	real enthalpyTotal = speedOfSoundSq / gammaMinusOne + .5f * velocitySq;

	real AlfvenSpeed = fabs(magneticField.x) / (sqrtDensity * sqrtVaccuumPermeability);
	real AlfvenSpeedSq = AlfvenSpeed * AlfvenSpeed;
	//Alfven speed is the absolute of the magnetic field in the normal direction -- to ensure ordering of eigenvalues (which might not be a necessary constraint)
	// however all my calculated results leave the eigenvalue as Bx, not |Bx|.  what to do?  enter sign(Bx).
	// if the eigenvalue was Bx and is now |Bx| then the corresponding eigenvector should be scaled by sign(Bx) ... (if it makes a difference at all)
	real sgnBx = magneticField.x >= 0.f ? 1.f : -1.f;
	
	real starSpeedSq = .5f * (speedOfSoundSq + magneticFieldSq / (density * vaccuumPermeability));
	real starSpeed = sqrt(starSpeedSq);
	real discr = starSpeedSq * starSpeedSq - speedOfSoundSq * AlfvenSpeedSq;
	real discrSqrt = sqrt(discr);
	real fastSpeedSq = starSpeedSq + discrSqrt;
	real fastSpeed = sqrt(fastSpeedSq);
	real slowSpeedSq = starSpeedSq - discrSqrt;
	real slowSpeed = sqrt(slowSpeedSq);
		
	//right eigenvectors
	//since these are the eigenvectors of the system wrt the symmetrized variables
	// the inverse of the right eigenvectors is the transpose of the right eigenvectors ... is the left eigenvectors
	real8 eigenvectorsWrtSymmetrized8[NUM_STATES];
	real* eigenvectorsWrtSymmetrized = (real*)eigenvectorsWrtSymmetrized8;

	//the eigenvectors wrt the symmetrizing variables are orthonormal, so the transpose is the inverse

	real alphaFast, alphaSlow;
	real4 lf = (real4)(0.f, 0.f, 0.f, 0.f);
	real4 ls = (real4)(0.f, 0.f, 0.f, 0.f);
	real4 mf = (real4)(0.f, 0.f, 0.f, 0.f);
	real4 ms = (real4)(0.f, 0.f, 0.f, 0.f);
	if (magneticFieldTLen < 1e-7f) {
		eigenvalues[0] = velocity.x - speedOfSound;
		eigenvalues[1] = velocity.x - AlfvenSpeed;
		eigenvalues[2] = velocity.x - AlfvenSpeed;
		eigenvalues[3] = velocity.x;
		eigenvalues[4] = velocity.x;
		eigenvalues[5] = velocity.x + AlfvenSpeed;
		eigenvalues[6] = velocity.x + AlfvenSpeed;
		eigenvalues[7] = velocity.x + speedOfSound;

		eigenvectorsWrtSymmetrized8[0] = (real8)(1.f, -1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f) * M_SQRT_1_2;
		eigenvectorsWrtSymmetrized8[1] = (real8)(0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f) * sgnBx * M_SQRT_1_2;
		eigenvectorsWrtSymmetrized8[2] = (real8)(0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f) * sgnBx * M_SQRT_1_2;
		eigenvectorsWrtSymmetrized8[3] = (real8)(0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f);
		eigenvectorsWrtSymmetrized8[4] = (real8)(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f);
		eigenvectorsWrtSymmetrized8[5] = (real8)(0.f, 0.f, 1.f, 0.f, 0.f, -1.f, 0.f, 0.f) * sgnBx * M_SQRT_1_2;
		eigenvectorsWrtSymmetrized8[6] = (real8)(0.f, 0.f, 0.f, 1.f, 0.f, 0.f, -1.f, 0.f) * sgnBx * M_SQRT_1_2;
		eigenvectorsWrtSymmetrized8[7] = (real8)(1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f) * M_SQRT_1_2;
	} else {
		eigenvalues[0] = velocity.x - fastSpeed;
		eigenvalues[1] = velocity.x - AlfvenSpeed;
		eigenvalues[2] = velocity.x - slowSpeed;
		eigenvalues[3] = velocity.x;
		eigenvalues[4] = velocity.x;
		eigenvalues[5] = velocity.x + slowSpeed;
		eigenvalues[6] = velocity.x + AlfvenSpeed;
		eigenvalues[7] = velocity.x + fastSpeed;
	
		//fast and slow eigenvectors are of the form [+/-k l +/-m 0]
		//Alfven eigenvectors are of the form [0 l -/+l 0]

		real4 l;
		if (magneticFieldTLen < 1e-7f) {
			l = (real4)(0.f, M_SQRT_1_2, 0.f, 0.f);
		} else {
			l = (real4)(0.f, magneticField.z, -magneticField.y, 0.f) * (M_SQRT_1_2 / magneticFieldTLen);
		}

		real kf = speedOfSound;
		lf = (real4)(fastSpeed, 0.f, 0.f, 0.f) - (fastSpeed * magneticField.x / (fastSpeedSq * vaccuumPermeability * density - magneticFieldXSq)) * magneticFieldT;
		mf = ((fastSpeedSq - speedOfSoundSq) * sqrtDensity * sqrtVaccuumPermeability / magneticFieldTSq) * magneticFieldT;
	
		real ks = speedOfSound;
		if (fabs(magneticField.x) < 1e-7f) {	//Bx=0 <=> ca=0 <=> cs=0, cf=c
			ls = magneticFieldT * (M_SQRT_2 * sqrtVaccuumPermeability * sqrtDensity * speedOfSound * starSpeed / magneticFieldTSq);
		} else {
			ls = (real4)(slowSpeed, 0.f, 0.f, 0.f) - (slowSpeed * magneticField.x / (slowSpeedSq * vaccuumPermeability * density - magneticFieldXSq)) * magneticFieldT;
		}
		ms = ((slowSpeedSq - speedOfSoundSq) * sqrtDensity * sqrtVaccuumPermeability / magneticFieldTSq) * magneticFieldT;

		//normalizing scalars for the fast and slow eigenvectors
		alphaFast = sqrt(dot(lf,lf) + dot(mf,mf) + kf*kf);
		alphaSlow = sqrt(dot(ls,ls) + dot(ms,ms) + ks*ks);

		//column-major (represented transposed)
		eigenvectorsWrtSymmetrized8[0] = (real8)(-kf,	lf.x, 	lf.y, 	lf.z, 	-mf.x, 	-mf.y, 	-mf.z,	0.f) / alphaFast; 
		eigenvectorsWrtSymmetrized8[1] = (real8)(0.f, 	l.x, 	l.y, 	l.z, 	l.x, 	l.y,	l.z,	0.f) * sgnBx;
		eigenvectorsWrtSymmetrized8[2] = (real8)(-ks, 	ls.x, 	ls.y, 	ls.z,	-ms.x, 	-ms.y,	-ms.z,	0.f) / alphaSlow;
		eigenvectorsWrtSymmetrized8[3] = (real8)(0.f, 	0.f, 	0.f, 	0.f, 	1.f, 	0.f, 	0.f,  	0.f);
		eigenvectorsWrtSymmetrized8[4] = (real8)(0.f, 	0.f, 	0.f, 	0.f, 	0.f, 	0.f, 	0.f,  	1.f);
		eigenvectorsWrtSymmetrized8[5] = (real8)(ks, 	ls.x, 	ls.y, 	ls.z, 	ms.x, 	ms.y, 	ms.z ,  0.f) / alphaSlow;
		eigenvectorsWrtSymmetrized8[6] = (real8)(0.f, 	l.x, 	l.y, 	l.z, 	-l.x, 	-l.y, 	-l.z ,  0.f) * sgnBx;
		eigenvectorsWrtSymmetrized8[7] = (real8)(kf, 	lf.x, 	lf.y,	lf.z, 	mf.x,	mf.y, 	mf.z ,  0.f) / alphaFast; 
	}

	//for all but the no-magnetic-field case transform the eigenvectors by dw/du

	//left and right eigenvectors above are of the flux derivative with respect to primitive variables
	//to find the eigenvectors of the flux with respect to the state variables, multiply by the derivative of the primitives with respect to the states
	//L = l * dw/du, R = du/dw * r
	//for l, r the left and right eigenvectors of derivative of flux wrt primitives
	//u = states, w = primitives
	//L, R the left and right eigenvectors of derivative of flux wrt state
	//this matches up with A = Q V Q^-1 = R V L = du/dw r V l dw/du

	//MBar
	real8 dCons_dSym8[8];	//column-major (represented transposed)
	real* dCons_dSym = (real*)dCons_dSym8;
	{
		real ctmp = speedOfSound * sqrtVaccuumPermeability / sqrtDensity; 
		real4 BBar = magneticField * (1.f / (sqrtDensity * sqrtVaccuumPermeability));
		dCons_dSym8[0] = (real8)(1.f,  velocity.x,  	velocity.y,  	velocity.z,  	0.f,	0.f,    0.f,    enthalpyTotal);
		dCons_dSym8[1] = (real8)(0.f,  speedOfSound,	0.f,            0.f,            0.f,	0.f,    0.f,    speedOfSound * velocity.x);
		dCons_dSym8[2] = (real8)(0.f,  0.f,            	speedOfSound,	0.f,            0.f,	0.f,    0.f,    speedOfSound * velocity.y);
		dCons_dSym8[3] = (real8)(0.f,  0.f,            	0.f,            speedOfSound,   0.f,	0.f,    0.f,    speedOfSound * velocity.z);
		dCons_dSym8[4] = (real8)(0.f,  0.f,            	0.f,            0.f,            ctmp,	0.f,    0.f,    speedOfSound * BBar.x);
		dCons_dSym8[5] = (real8)(0.f,  0.f,            	0.f,            0.f,            0.f,	ctmp,	0.f,    speedOfSound * BBar.y);
		dCons_dSym8[6] = (real8)(0.f,  0.f,            	0.f,            0.f,            0.f,	0.f,    ctmp,	speedOfSound * BBar.z);
		dCons_dSym8[7] = (real8)(-1.f, -velocity.x,		-velocity.y,	-velocity.z,	0.f,	0.f,    0.f,    -.5f * velocitySq);
	}
	
	//MBar^-1
	real8 dSym_dCons8[8];	//column-major (represented transposed)
	real* dSym_dCons = (real*)dSym_dCons8;
	{
		real gammaBar = gammaMinusOne / speedOfSoundSq;
		real4 negVGammaBar = -gammaBar * velocity;
		real4 Btmp = magneticField * (-gammaBar / vaccuumPermeability);
		real oneOverC = 1.f / speedOfSound;
		real tmpc = sqrtDensity / (sqrtVaccuumPermeability * speedOfSound);
		dSym_dCons8[0] = (real8)(.5f * gammaBar * velocitySq,	-velocity.x * oneOverC,	-velocity.y * oneOverC,	-velocity.z * oneOverC, 0.f, 	0.f,	0.f,	gammaBar * (velocitySq - enthalpyTotal));
		dSym_dCons8[1] = (real8)(negVGammaBar.x,				oneOverC,				0.f,					0.f, 					0.f,	0.f,	0.f,	negVGammaBar.x);
		dSym_dCons8[2] = (real8)(negVGammaBar.y,				0.f,					oneOverC,				0.f,					0.f,	0.f,	0.f,	negVGammaBar.y);
		dSym_dCons8[3] = (real8)(negVGammaBar.z, 				0.f, 					0.f, 					oneOverC,				0.f, 	0.f,	0.f,	negVGammaBar.z);
		dSym_dCons8[4] = (real8)(Btmp.x, 						0.f, 					0.f, 					0.f, 					tmpc, 	0.f, 	0.f,	Btmp.x);
		dSym_dCons8[5] = (real8)(Btmp.y, 						0.f, 					0.f, 					0.f, 					0.f,	tmpc, 	0.f,	Btmp.y);
		dSym_dCons8[6] = (real8)(Btmp.z,						0.f, 					0.f, 					0.f, 					0.f, 	0.f, 	tmpc,	Btmp.z);
		dSym_dCons8[7] = (real8)(gammaBar, 						0.f,					0.f,					0.f,					0.f,	0.f,	0.f,	gammaBar);
	}

	//R = dCons/dSym * r <=> R_i = [dCons/dSym]_ik * r_k <=> R_ij = [dCons/dSym]_ik * r_kj
	//L = l * dSym/dCons <=> L_j = l_k * [dSym/dCons]_kj <=> L_ij = l_ik * [dSym/dCons]_kj
	//A = R * Lambda * L
	for (int i = 0; i < NUM_STATES; ++i) {
		for (int j = 0; j < NUM_STATES; ++j) {
			real sum;
			
			sum = 0.f;
			for (int k = 0; k < NUM_STATES; ++k) {
				sum += eigenvectorsWrtSymmetrized[k + NUM_STATES * i] * dSym_dCons[k + NUM_STATES * j];	//left a_ik == right a_ki
			}
			eigenvectorsInverse[i + NUM_STATES * j] = sum;
			
			sum = 0.f;
			for (int k = 0; k < NUM_STATES; ++k) {
				sum += dCons_dSym[i + NUM_STATES * k] * eigenvectorsWrtSymmetrized[k + NUM_STATES * j];
			}
			eigenvectors[i + NUM_STATES * j] = sum;
		}
	}

#ifdef DEBUG_OUTPUT
	if (index == DEBUG_INDEX) {
		printf("side %d\n", side);
		printf("i %d\n", index);
		//heart of current problem: magnetic energy density is exceeding our total energy density
		// so the K+P energy density comes out negative ...
		//magnetic energy density comes from the magnetic field states
		//total energy density comes from the the ENERGY_TOTAL state
		// this means our eigenvectors are contributing less to total energy than they should be. 
		printf("density %f\n", density);
		printf("sqrt(density) %f\n", sqrtDensity);
		printf("vaccuum permeability %f\n", vaccuumPermeability);
		printf("magnetic field %f %f %f\n", magneticField.x, magneticField.y, magneticField.z);
		printf("magnetic field T %f %f %f\n", magneticFieldT.x, magneticFieldT.y, magneticFieldT.z);
		printf("magnetic field T length %f\n", magneticFieldTLen);
		printf("magnetic field T length^2 %f\n", magneticFieldTSq);
		printf("lf %f %f %f\n", lf.x, lf.y, lf.z);
		printf("ls %f %f %f\n", ls.x, ls.y, ls.z);
		printf("mf %f %f %f\n", mf.x, mf.y, mf.z);
		printf("ms %f %f %f\n", ms.x, ms.y, ms.z);
		printf("gamma %f\n", gamma);
		printf("pressure %f\n", pressure);
		printf("speedOfSound %f\n", speedOfSound);
		printf("speedOfSoundSq %f\n", speedOfSoundSq);
		printf("fastSpeed %f\n", fastSpeed);
		printf("fastSpeedSq %f\n", fastSpeedSq);
		printf("AlfvenSpeed %f\n", AlfvenSpeed);
		printf("AlfvenSpeedSq %f\n", AlfvenSpeedSq);
		printf("slowSpeed %f\n", slowSpeed);
		printf("slowSpeedSq %f\n", slowSpeedSq);
		printf("alphaFast %f\n", alphaFast);
		printf("alphaSlow %f\n", alphaSlow);
		printf("eigenvalues");
		for (int i = 0; i < NUM_STATES; ++i) {
			printf(" %f", eigenvalues[i]);
		}
		printf("\n");
		printf("symmetrized eigenvectors\n");
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				printf(" %f", eigenvectorsWrtSymmetrized[i + NUM_STATES * j]);
			}
			printf("\n");
		}
		printf("symmetrized eigenvector orthogonality\n");
		real totalError = 0.f;
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				real sum = 0.f;
				for (int k = 0; k < NUM_STATES; ++k) {
					sum += eigenvectorsWrtSymmetrized[k + NUM_STATES * i] * eigenvectorsWrtSymmetrized[k + NUM_STATES * j];	//left i,k * right k,j == right k,i * right k,j
				}
				printf(" %f", sum);
				totalError += fabs(sum - (i == j ? 1.f : 0.f));
			}
			printf("\n");
		}
		printf("symmetrized eigenvector error %f\n", totalError);
		printf("side %d\n", side);
		printf("i %d\n", index);
		printf("conservative eigenvector orthogonality\n");
		totalError = 0.f;
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				real sum = 0.f;
				for (int k = 0; k < NUM_STATES; ++k) {
					sum += eigenvectorsInverse[i + NUM_STATES * k] * eigenvectors[k + NUM_STATES * j];
				}
				printf(" %f", sum);
				totalError += fabs(sum - (i == j ? 1.f : 0.f));
			}
			printf("\n");
		}
		printf("conservative eigenvector error %f\n", totalError);
		printf("dCons/dSym\n");
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				printf(" %f", dCons_dSym[i + NUM_STATES * j]);
			}
			printf("\n");
		}
		printf("dSym/dCons\n");
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				printf(" %f", dSym_dCons[i + NUM_STATES * j]);
			}
			printf("\n");
		}
		printf("dCons/dSym_ik * dSym/dCons_kj orthogonality\n");
		totalError = 0.f;
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				real sum = 0.f;
				for (int k = 0; k < NUM_STATES; ++k) {
					sum += dCons_dSym[i + NUM_STATES * k] * dSym_dCons[k + NUM_STATES * j];
				}
				printf(" %f", sum);
				totalError += fabs(sum - (i == j ? 1.f : 0.f));
			}
			printf("\n");
		}
		printf("dCons/dSym_ik * dSym/dCons_kj error %f\n", totalError);
	}
#endif	//DEBUG_OUTPUT

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
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = -eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y] = tmp;
			
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X] = -eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Y] = tmp;
			
			//a -90' rotation applied to the RHS of A must be corrected with a 90' rotation on the LHS of A
			//this rotates the elements of the column vectors by 90'
			//each column's x <- y, y <- -x
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = -eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i] = tmp;
			
			tmp = eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i] = -eigenvectors[STATE_MAGNETIC_FIELD_Y + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = -eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z] = tmp;
			
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_X] = -eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_FIELD_Z] = tmp;
			
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = -eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i] = tmp;
			
			tmp = eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_X + NUM_STATES * i] = -eigenvectors[STATE_MAGNETIC_FIELD_Z + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_FIELD_Z + NUM_STATES * i] = tmp;
		}
	}
#endif
#endif
	

}

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer)
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

	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 0);
#if DIM > 1
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 1);
#endif
#if DIM > 2
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 2);
#endif
}


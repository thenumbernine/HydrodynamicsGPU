/*
using the following:
"Eigenvalues, Eigenvectors, and Symmetrization of the Magneto-Hydrodynamic (MHD) Equations" by Jameson 2006

have dug through
"A Numerical Solution of Hyperbolic Partial Differential Equations", Trangenstein, 2007
"A multidimensional upwind scheme for magnetohydrodynamics" by Falle, Komissarov, Joarder, 1998
"A Solution-Adaptive Upwind Scheme for Ideal Magnetohydrodynamics" by Powell, Roe, Linde, Gombosi, Zeeuw, 1999
*/

#include "HydroGPU/Shared/Common.h"

#define M_SQRT_1_2	0.7071067811865475727373109293694142252206802368164f

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
	real enthalpyTotalL = (totalHydroEnergyDensityL + pressureL) / densityL;	//should enthalpy total also include magnetic energy?
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
	real enthalpyTotalR = (totalHydroEnergyDensityR + pressureR) / densityR;
	real roeWeightR = .5f;//sqrt(densityR);

	//3.5.2 "In this paper, a simple arithmetic averaging of the primitive variables is done to compute the interface state."
	//but while I'm solving the degeneracy case, I'll use Roe weighting
	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real4 velocity = (velocityL * roeWeightL + velocityR * roeWeightR) * roeWeightNormalization;
	real4 magneticField = (magneticFieldL * roeWeightL + magneticFieldR * roeWeightR) * roeWeightNormalization;
	real pressure = (pressureL * roeWeightL + pressureR * roeWeightR) * roeWeightNormalization;
	real enthalpyTotal = (enthalpyTotalL * roeWeightL + enthalpyTotalR * roeWeightR) * roeWeightNormalization;
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

	real sqrtDensity = sqrt(density);
	real sqrtRhoMu = sqrtDensity * sqrtVaccuumPermeability;
	real oneOverSqrtRhoMu = 1.f / sqrtRhoMu;
	real4 BBar = magneticField * oneOverSqrtRhoMu;
	real BBarXSq = BBar.x * BBar.x;
	real4 BBarT = real4(0.f, BBar.y, BBar.z, 0.f);	//magnetic component perpendicular to normal (which is the x axis)
	real BBarTSq = BBar.y * BBar.y + BBar.z * BBar.z;
	real BBarTLen = sqrt(BBarTSq);
	real BBarSq = BBarXSq + BBarTSq;

	//matrices are stored as A_ij = A[i + height * j]

	real speedOfSound = sqrt(gamma * pressure / density);
	real speedOfSoundSq = speedOfSound * speedOfSound;

	real AlfvenSpeed = BBar.x;
	real AlfvenSpeedSq = AlfvenSpeed * AlfvenSpeed;
	real starSpeedSq = .5f * (speedOfSoundSq + BBarSq);
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

#ifdef DEBUG_OUTPUT
if (index == DEBUG_INDEX) {
printf("magnetic field n=0\n");
}
#endif	//DEBUG_OUTPUT

	//eigenvalues
	eigenvalues[0] = velocity.x - fastSpeed;
	eigenvalues[1] = velocity.x - AlfvenSpeed;
	eigenvalues[2] = velocity.x - slowSpeed;
	eigenvalues[3] = velocity.x;
	eigenvalues[4] = velocity.x;
	eigenvalues[5] = velocity.x + slowSpeed;
	eigenvalues[6] = velocity.x + AlfvenSpeed;
	eigenvalues[7] = velocity.x + fastSpeed;

	//the eigenvectors wrt the symmetrizing variables are orthonormal, so the transpose is the inverse

	real4 l;
	if (BBarTSq < 1e-10f) {
		//l = (real4)(0.f, M_SQRT_1_2, 0.f, 0.f);
		l.x = 0.f;
		l.y = M_SQRT_1_2;
		l.z = 0.f;
		l.w = 0.f;
	} else {
		//l = (real4)(0.f, BBar.z, -BBar.y, 0.f) * M_SQRT_1_2 / BBarTLen;
		l.x = 0.f;
		l.y = BBar.z / BBarTLen * M_SQRT_1_2;
		l.z = -BBar.y / BBarTLen * M_SQRT_1_2;
		l.w = 0.f;
	}
	
	real4 lf = fastSpeed * ((real4)(1.f, 0.f, 0.f, 0.f) - BBar.x / (fastSpeedSq - BBarXSq) * BBarT);
	real4 mf = fastSpeedSq / (fastSpeedSq - BBarXSq) * BBarT;
	real4 ls = slowSpeed * ((real4)(1.f, 0.f, 0.f, 0.f) - BBar.x / (slowSpeedSq - BBarXSq) * BBarT);
	real4 ms = slowSpeedSq / (slowSpeedSq - BBarXSq) * BBarT;

	real alphaFast = sqrt(dot(lf,lf) + dot(mf,mf) + speedOfSoundSq);
	real alphaSlow = sqrt(dot(ls,ls) + dot(ms,ms) + speedOfSoundSq);

	//column-major (represented transposed)
	eigenvectorsWrtSymmetrized8[0] = (real8)(-speedOfSound, lf.x, lf.y, lf.z, 0.f, -mf.x, -mf.y, -mf.z) / alphaFast; 
	eigenvectorsWrtSymmetrized8[1] = (real8)(0.f, l.x, l.y, l.z, 0.f, l.x, l.y, l.z);
	eigenvectorsWrtSymmetrized8[2] = (real8)(-speedOfSound, ls.x, ls.y, ls.z, 0.f, -ms.x, -ms.y, -ms.z) / alphaSlow;
	eigenvectorsWrtSymmetrized8[3] = (real8)(0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f);
	eigenvectorsWrtSymmetrized8[4] = (real8)(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f);
	eigenvectorsWrtSymmetrized8[5] = (real8)(speedOfSound, ls.x, ls.y, ls.z, 0.f, ms.x, ms.y, ms.z) / alphaSlow;
	eigenvectorsWrtSymmetrized8[6] = (real8)(0.f, l.x, l.y, l.z, 0.f, -l.x, -l.y, -l.z);
	eigenvectorsWrtSymmetrized8[7] = (real8)(speedOfSound, lf.x, lf.y, lf.z, 0.f, mf.x, mf.y, mf.z) / alphaFast; 
	
	//for all but the no-magnetic-field case transform the eigenvectors by dw/du

	//left and right eigenvectors above are of the flux derivative with respect to primitive variables
	//to find the eigenvectors of the flux with respect to the state variables, multiply by the derivative of the primitives with respect to the states
	//L = l * dw/du, R = du/dw * r
	//for l, r the left and right eigenvectors of derivative of flux wrt primitives
	//u = states, w = primitives
	//L, R the left and right eigenvectors of derivative of flux wrt state
	//this matches up with A = Q V Q^-1 = R V L = du/dw r V l dw/du

	real rhoOverC = density / speedOfSound;
	
	//in the absense of a magnetic field, we get MBar * Mbar^-1 = diag(1/4, 1,1,1,1,1,1, 1/4)

	//MBar
	real8 dCons_dPrim8[8];	//column-major (represented transposed)
	dCons_dPrim8[0] = (real8)(rhoOverC,  		rhoOverC * velocity.x,  rhoOverC * velocity.y,  rhoOverC * velocity.z,  0.f,       0.f,       0.f,       rhoOverC * enthalpyTotal);
	dCons_dPrim8[1] = (real8)(0.f,       		density,                0.f,                    0.f,                    0.f,       0.f,       0.f,       density * velocity.x);
	dCons_dPrim8[2] = (real8)(0.f,       		0.f,                    density,                0.f,                    0.f,       0.f,       0.f,       density * velocity.y);
	dCons_dPrim8[3] = (real8)(0.f,       		0.f,                    0.f,                    density,                0.f,       0.f,       0.f,       density * velocity.y);
	dCons_dPrim8[4] = (real8)(-rhoOverC, 		-rhoOverC * velocity.x, -rhoOverC * velocity.y, -rhoOverC * velocity.z, 0.f,       0.f,       0.f,       -.5f * rhoOverC * velocitySq);
	dCons_dPrim8[5] = (real8)(0.f,       		0.f,                    0.f,                    0.f,                    sqrtRhoMu, 0.f,       0.f,       density * BBar.x);
	dCons_dPrim8[6] = (real8)(0.f,       		0.f,                    0.f,                    0.f,                    0.f,       sqrtRhoMu, 0.f,       density * BBar.y);
	dCons_dPrim8[7] = (real8)(0.f,       		0.f,                    0.f,                    0.f,                    0.f,       0.f,       sqrtRhoMu, density * BBar.z);
	real* dCons_dPrim = (real*)dCons_dPrim8;

	real gammaBar = gammaMinusOne / (density * speedOfSound);
	real4 Btmp = -gammaBar * magneticField / vaccuumPermeability;
	
	//MBar^-1
	real8 dPrim_dCons8[8];	//column-major (represented transposed)
	dPrim_dCons8[0] = (real8)(gammaBar * velocitySq / speedOfSound, -velocity.x / density, -velocity.y / density, 	-velocity.z / density, 	gammaBar * (velocitySq - enthalpyTotal), 0.f, 				0.f, 				0.f);
	dPrim_dCons8[1] = (real8)(-velocity.x * gammaBar,				1.f / density,			0.f,					0.f, 					-gammaBar * velocity.x,					0.f,				0.f,				0.f);
	dPrim_dCons8[2] = (real8)(-velocity.y * gammaBar,				0.f,					1.f / density,			0.f,					-gammaBar * velocity.y,					0.f,				0.f,				0.f);
	dPrim_dCons8[3] = (real8)(-velocity.z * gammaBar, 				0.f, 					0.f, 					1.f / density, 			-gammaBar * velocity.z, 				0.f, 				0.f,				0.f);
	dPrim_dCons8[4] = (real8)(-Btmp.x, 								0.f, 					0.f, 					0.f, 					-Btmp.x, 								oneOverSqrtRhoMu, 	0.f, 				0.f);
	dPrim_dCons8[5] = (real8)(-Btmp.y, 								0.f, 					0.f, 					0.f, 					-Btmp.y,								0.f,				oneOverSqrtRhoMu, 	0.f);
	dPrim_dCons8[6] = (real8)(-Btmp.z,				 				0.f, 					0.f, 					0.f, 					-Btmp.z,								0.f, 				0.f, 				oneOverSqrtRhoMu);
	dPrim_dCons8[7] = (real8)(gammaBar, 							0.f,					0.f,					0.f,					gammaBar,								0.f,				0.f,				0.f);
	real* dPrim_dCons = (real*)dPrim_dCons8;

	//R = dCons/dPrim * r <=> R_i = [dCons/dPrim]_ik * r_k <=> R_ij = [dCons/dPrim]_ik * r_kj
	//L = l * dPrim/dCons <=> L_j = l_k * [dPrim/dCons]_kj <=> L_ij = l_ik * [dPrim/dCons]_kj
	//A = R * Lambda * L
	for (int i = 0; i < NUM_STATES; ++i) {
		for (int j = 0; j < NUM_STATES; ++j) {
			real sum;
			
			sum = 0.f;
			for (int k = 0; k < NUM_STATES; ++k) {
				sum += eigenvectorsWrtSymmetrized[k + NUM_STATES * i] * dPrim_dCons[k + NUM_STATES * j];	//left a_ik == right a_ki
			}
			eigenvectorsInverse[i + NUM_STATES * j] = sum;
			
			sum = 0.f;
			for (int k = 0; k < NUM_STATES; ++k) {
				sum += dCons_dPrim[i + NUM_STATES * k] * eigenvectorsWrtSymmetrized[k + NUM_STATES * j];
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
		printf("B-bar %f %f %f\n", BBar.x, BBar.y, BBar.z);
		printf("B-bar T %f %f %f\n", BBarT.x, BBarT.y, BBarT.z);
		printf("B-bar T length %f\n", BBarTLen);
		printf("B-bar T length^2 %f\n", BBarTSq);
		printf("l %f %f %f %f\n", l.x, l.y, l.z, l.w);
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
		printf("dCons/dPrim\n");
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				printf(" %f", dCons_dPrim[i + NUM_STATES * j]);
			}
			printf("\n");
		}
		printf("dPrim/dCons\n");
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				printf(" %f", dPrim_dCons[i + NUM_STATES * j]);
			}
			printf("\n");
		}
		printf("dCons/dPrim_ik * dPrim/dCons_kj orthogonality\n");
		totalError = 0.f;
		for (int i = 0; i < NUM_STATES; ++i) {
			for (int j = 0; j < NUM_STATES; ++j) {
				real sum = 0.f;
				for (int k = 0; k < NUM_STATES; ++k) {
					sum += dCons_dPrim[i + NUM_STATES * k] * dPrim_dCons[k + NUM_STATES * j];
				}
				printf(" %f", sum);
				totalError += fabs(sum - (i == j ? 1.f : 0.f));
			}
			printf("\n");
		}
		printf("dCons/dPrim_ik * dPrim/dCons_kj error %f\n", totalError);
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


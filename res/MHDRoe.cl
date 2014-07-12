/*
using "A Solution-Adaptive Upwind Scheme for Ideal Magnetohydrodynamics" by Powell, Roe, Linde, Gombosi, Zeeuw, 1999 
*/

#include "HydroGPU/Shared/Common.h"

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
	
	const real gammaMinusOne = GAMMA - 1.f;

	real densityL = stateL[STATE_DENSITY];
	real invDensityL = 1.f / densityL;
	real4 velocityL = VELOCITY(stateL);
	real4 magneticFieldL = (real4)(stateL[STATE_MAGNETIC_FIELD_X], stateL[STATE_MAGNETIC_FIELD_Y], stateL[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticEnergyDensityL = .5f * dot(magneticFieldL, magneticFieldL) / MU0;
	real totalPlasmaEnergyDensityL = stateL[STATE_ENERGY_TOTAL];
	real totalHydroEnergyDensityL = totalPlasmaEnergyDensityL - magneticEnergyDensityL;
	real kineticEnergyDensityL = .5f * densityL * dot(velocityL, velocityL);
	real potentialEnergyDensityL = densityL * potentialBuffer[indexPrev];
	real internalEnergyDensityL = totalHydroEnergyDensityL - kineticEnergyDensityL - potentialEnergyDensityL;
	real pressureL = gammaMinusOne * internalEnergyDensityL;

	real densityR = stateR[STATE_DENSITY];
	real invDensityR = 1.f / densityR;
	real4 velocityR = VELOCITY(stateR);
	real4 magneticFieldR = (real4)(stateR[STATE_MAGNETIC_FIELD_X], stateR[STATE_MAGNETIC_FIELD_Y], stateR[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticEnergyDensityR = .5f * dot(magneticFieldR, magneticFieldR) / MU0;
	real totalPlasmaEnergyDensityR = stateR[STATE_ENERGY_TOTAL];
	real totalHydroEnergyDensityR = totalPlasmaEnergyDensityR - magneticEnergyDensityR;
	real kineticEnergyDensityR = .5f * densityR * dot(velocityR, velocityR);
	real potentialEnergyDensityR = densityR * potentialBuffer[index];
	real internalEnergyDensityR = totalHydroEnergyDensityR - kineticEnergyDensityR - potentialEnergyDensityR;
	real pressureR = gammaMinusOne * internalEnergyDensityR;

	//3.5.2 "In this paper, a simple arithmetic averaging of the primitive variables is done to compute the interface state."
	real density = .5f * (densityL + densityR);
	real4 velocity = .5f * (velocityL + velocityR);
	real4 magneticField = .5f * (magneticFieldL + magneticFieldR);
	real pressure = .5f * (pressureL * pressureR);
	
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
	real speedOfSound = sqrt(max(0.f, GAMMA * pressure / density));
	real speedOfSoundSq = speedOfSound * speedOfSound;
	real magneticFieldSq = dot(magneticField, magneticField);
	real magneticFieldXSq = magneticField.x * magneticField.x;
	
	real AlfvenSpeed = magneticField.x / sqrtDensity;
	real AlfvenSpeedSq = AlfvenSpeed * AlfvenSpeed;
	//TODO update slow and fast speeds
	real tmp1 = (GAMMA * pressure + magneticFieldSq) / density;
	real discr = max(0.f, tmp1 * tmp1 - 4.f * GAMMA * pressure * magneticFieldXSq / (density * density));
	real tmp2 = sqrt(discr);
	real fastSpeedSq = max(.5f * tmp1 + tmp2, 0.f);
	real fastSpeed = sqrt(fastSpeedSq);
	real slowSpeedSq = max(.5f * tmp1 - tmp2, 0.f);
	real slowSpeed = sqrt(slowSpeedSq);

	//TODO define
	real oneOverFastSpeedSqMinusSlowSpeedSq = 1.f / (fastSpeedSq - slowSpeedSq);
	real alphaFast = sqrt(max(0.f, (speedOfSoundSq - slowSpeedSq) * oneOverFastSpeedSqMinusSlowSpeedSq));
	real alphaSlow = sqrt(max(0.f, (fastSpeedSq - speedOfSoundSq) * oneOverFastSpeedSqMinusSlowSpeedSq));
	real oneOverMagneticFieldYZLen = 1.f / sqrt(magneticField.y * magneticField.y + magneticField.z * magneticField.z);
	real betaY = magneticField.y * oneOverMagneticFieldYZLen;
	real betaZ = magneticField.z * oneOverMagneticFieldYZLen;
	real sgnBx;
	if (magneticField.x > 0.f) {
		sgnBx = 1.f;
	} else {
		sgnBx = -1.f;
	}

	//eigenvalues

	eigenvalues[0] = velocity.x - fastSpeed;
	eigenvalues[1] = velocity.x - magneticField.x / density;	//= AlfvenSpeed / sqrtDensity
	eigenvalues[2] = velocity.x - slowSpeed;
	eigenvalues[3] = velocity.x;
	eigenvalues[4] = velocity.x;
	eigenvalues[5] = velocity.x + slowSpeed;
	eigenvalues[6] = velocity.x + magneticField.x / density;
	eigenvalues[7] = velocity.x + fastSpeed;

	//eigenvectors
	//stored as A_ij = A[i + height * j]

	real eigenvectorsWrtPrimitives[NUM_STATES * NUM_STATES];

#define M_SQRT_1_2	0.7071067811865475727373109293694142252206802368164f

	//fast magnetoacoustic col 
	eigenvectorsWrtPrimitives[0 + NUM_STATES * 0] = density * alphaFast;
	eigenvectorsWrtPrimitives[1 + NUM_STATES * 0] = -alphaFast * fastSpeed;
	eigenvectorsWrtPrimitives[2 + NUM_STATES * 0] = alphaSlow * slowSpeed * betaY * sgnBx;
	eigenvectorsWrtPrimitives[3 + NUM_STATES * 0] = alphaSlow * slowSpeed * betaY * sgnBx;
	eigenvectorsWrtPrimitives[4 + NUM_STATES * 0] = 0.f;
	eigenvectorsWrtPrimitives[5 + NUM_STATES * 0] = alphaSlow * sqrtDensity * speedOfSound * betaY;
	eigenvectorsWrtPrimitives[6 + NUM_STATES * 0] = alphaSlow * sqrtDensity * speedOfSound * betaZ;
	eigenvectorsWrtPrimitives[7 + NUM_STATES * 0] = alphaFast * GAMMA * pressure;
	//Alfven col
	eigenvectorsWrtPrimitives[0 + NUM_STATES * 1] = 0.f;
	eigenvectorsWrtPrimitives[1 + NUM_STATES * 1] = 0.f;
	eigenvectorsWrtPrimitives[2 + NUM_STATES * 1] = -betaZ * M_SQRT_1_2;
	eigenvectorsWrtPrimitives[3 + NUM_STATES * 1] = betaY * M_SQRT_1_2;
	eigenvectorsWrtPrimitives[4 + NUM_STATES * 1] = 0.f;
	eigenvectorsWrtPrimitives[5 + NUM_STATES * 1] = -sqrtDensity * M_SQRT_1_2 * betaZ;
	eigenvectorsWrtPrimitives[6 + NUM_STATES * 1] = sqrtDensity * M_SQRT_1_2 * betaY;
	eigenvectorsWrtPrimitives[7 + NUM_STATES * 1] = 0.f;
	//slow magnetoacoustic col
	eigenvectorsWrtPrimitives[0 + NUM_STATES * 2] = density * alphaSlow;
	eigenvectorsWrtPrimitives[1 + NUM_STATES * 2] = -alphaSlow * slowSpeed;
	eigenvectorsWrtPrimitives[2 + NUM_STATES * 2] = -alphaFast * fastSpeed * betaY * sgnBx;
	eigenvectorsWrtPrimitives[3 + NUM_STATES * 2] = -alphaFast * fastSpeed * betaZ * sgnBx;
	eigenvectorsWrtPrimitives[4 + NUM_STATES * 2] = 0.f;
	eigenvectorsWrtPrimitives[5 + NUM_STATES * 2] = -alphaFast * sqrtDensity * speedOfSound * betaY;
	eigenvectorsWrtPrimitives[6 + NUM_STATES * 2] = -alphaFast * sqrtDensity * speedOfSound * betaZ;
	eigenvectorsWrtPrimitives[7 + NUM_STATES * 2] = alphaSlow * GAMMA * pressure;
	//entropy col
	eigenvectorsWrtPrimitives[0 + NUM_STATES * 3] = 1.f;
	eigenvectorsWrtPrimitives[1 + NUM_STATES * 3] = 0.f; 
	eigenvectorsWrtPrimitives[2 + NUM_STATES * 3] = 0.f;
	eigenvectorsWrtPrimitives[3 + NUM_STATES * 3] = 0.f;
	eigenvectorsWrtPrimitives[4 + NUM_STATES * 3] = 0.f;
	eigenvectorsWrtPrimitives[5 + NUM_STATES * 3] = 0.f;
	eigenvectorsWrtPrimitives[6 + NUM_STATES * 3] = 0.f;
	eigenvectorsWrtPrimitives[7 + NUM_STATES * 3] = 0.f;	
	//divergence col 
	eigenvectorsWrtPrimitives[0 + NUM_STATES * 4] = 0.f;
	eigenvectorsWrtPrimitives[1 + NUM_STATES * 4] = 0.f;
	eigenvectorsWrtPrimitives[2 + NUM_STATES * 4] = 0.f;
	eigenvectorsWrtPrimitives[3 + NUM_STATES * 4] = 0.f;
	eigenvectorsWrtPrimitives[4 + NUM_STATES * 4] = 1.f;
	eigenvectorsWrtPrimitives[5 + NUM_STATES * 4] = 0.f;
	eigenvectorsWrtPrimitives[6 + NUM_STATES * 4] = 0.f;
	eigenvectorsWrtPrimitives[7 + NUM_STATES * 4] = 0.f;
	//slow magnetoacoustic col
	eigenvectorsWrtPrimitives[0 + NUM_STATES * 5] = density * alphaSlow;
	eigenvectorsWrtPrimitives[1 + NUM_STATES * 5] = alphaSlow * slowSpeed;
	eigenvectorsWrtPrimitives[2 + NUM_STATES * 5] = alphaFast * fastSpeed * betaY * sgnBx;
	eigenvectorsWrtPrimitives[3 + NUM_STATES * 5] = alphaFast * fastSpeed * betaZ * sgnBx;
	eigenvectorsWrtPrimitives[4 + NUM_STATES * 5] = 0.f;
	eigenvectorsWrtPrimitives[5 + NUM_STATES * 5] = -alphaFast * sqrtDensity * speedOfSound * betaY;
	eigenvectorsWrtPrimitives[6 + NUM_STATES * 5] = -alphaFast * sqrtDensity * speedOfSound * betaZ;
	eigenvectorsWrtPrimitives[7 + NUM_STATES * 5] = alphaSlow * GAMMA * pressure;
	//Alfven col
	eigenvectorsWrtPrimitives[0 + NUM_STATES * 6] = 0.f;
	eigenvectorsWrtPrimitives[1 + NUM_STATES * 6] = 0.f;
	eigenvectorsWrtPrimitives[2 + NUM_STATES * 6] = -betaZ * M_SQRT_1_2;
	eigenvectorsWrtPrimitives[3 + NUM_STATES * 6] = betaY * M_SQRT_1_2;
	eigenvectorsWrtPrimitives[4 + NUM_STATES * 6] = 0.f;
	eigenvectorsWrtPrimitives[5 + NUM_STATES * 6] = sqrtDensity * M_SQRT_1_2 * betaZ;
	eigenvectorsWrtPrimitives[6 + NUM_STATES * 6] = -sqrtDensity * M_SQRT_1_2 * betaY;
	eigenvectorsWrtPrimitives[7 + NUM_STATES * 6] = 0.f;
	//fast magnetoacoustic col
	eigenvectorsWrtPrimitives[0 + NUM_STATES * 7] = density * alphaFast;
	eigenvectorsWrtPrimitives[1 + NUM_STATES * 7] = alphaFast * fastSpeed;
	eigenvectorsWrtPrimitives[2 + NUM_STATES * 7] = -alphaSlow * slowSpeed * betaY * sgnBx;
	eigenvectorsWrtPrimitives[3 + NUM_STATES * 7] = -alphaSlow * slowSpeed * betaY * sgnBx;
	eigenvectorsWrtPrimitives[4 + NUM_STATES * 7] = 0.f;
	eigenvectorsWrtPrimitives[5 + NUM_STATES * 7] = alphaSlow * sqrtDensity * speedOfSound * betaY;
	eigenvectorsWrtPrimitives[6 + NUM_STATES * 7] = alphaSlow * sqrtDensity * speedOfSound * betaZ;
	eigenvectorsWrtPrimitives[7 + NUM_STATES * 7] = alphaFast * GAMMA * pressure;


	//eigenvectors inverse
	real eigenvectorsInverseWrtPrimitives[NUM_STATES * NUM_STATES];
	//fast magnetoacoustic row
	eigenvectorsInverseWrtPrimitives[0 + NUM_STATES * 0] = 0.f;
	eigenvectorsInverseWrtPrimitives[0 + NUM_STATES * 1] = -alphaFast * fastSpeed / (2.f * speedOfSoundSq);
	eigenvectorsInverseWrtPrimitives[0 + NUM_STATES * 2] = alphaSlow * slowSpeed * betaY * sgnBx / (2.f * speedOfSoundSq);
	eigenvectorsInverseWrtPrimitives[0 + NUM_STATES * 3] = alphaSlow * slowSpeed * betaZ * sgnBx / (2.f * speedOfSoundSq);
	eigenvectorsInverseWrtPrimitives[0 + NUM_STATES * 4] = 0.f;
	eigenvectorsInverseWrtPrimitives[0 + NUM_STATES * 5] = alphaSlow * betaY / (2.f * sqrtDensity * speedOfSound);
	eigenvectorsInverseWrtPrimitives[0 + NUM_STATES * 6] = alphaSlow * betaZ / (2.f * sqrtDensity * speedOfSound);
	eigenvectorsInverseWrtPrimitives[0 + NUM_STATES * 7] = alphaFast / (2.f * density * speedOfSoundSq);
	//Alfven row
	eigenvectorsInverseWrtPrimitives[1 + NUM_STATES * 0] = 0.f;
	eigenvectorsInverseWrtPrimitives[1 + NUM_STATES * 1] = 0.f;
	eigenvectorsInverseWrtPrimitives[1 + NUM_STATES * 2] = -betaZ / sqrtDensity;
	eigenvectorsInverseWrtPrimitives[1 + NUM_STATES * 3] = betaY / sqrtDensity;
	eigenvectorsInverseWrtPrimitives[1 + NUM_STATES * 4] = 0.f;
	eigenvectorsInverseWrtPrimitives[1 + NUM_STATES * 5] = -betaZ * M_SQRT_1_2 / sqrtDensity;
	eigenvectorsInverseWrtPrimitives[1 + NUM_STATES * 6] = betaY * M_SQRT_1_2 / sqrtDensity;
	eigenvectorsInverseWrtPrimitives[1 + NUM_STATES * 7] = 0.f;
	//slow magnetoacoustic row
	eigenvectorsInverseWrtPrimitives[2 + NUM_STATES * 0] = 0.f;
	eigenvectorsInverseWrtPrimitives[2 + NUM_STATES * 1] = -alphaSlow * slowSpeed * .5f / speedOfSoundSq;
	eigenvectorsInverseWrtPrimitives[2 + NUM_STATES * 2] = -alphaFast * fastSpeed * betaY * sgnBx * .5f / speedOfSoundSq;
	eigenvectorsInverseWrtPrimitives[2 + NUM_STATES * 3] = -alphaFast * fastSpeed * betaZ * sgnBx * .5f / speedOfSoundSq;
	eigenvectorsInverseWrtPrimitives[2 + NUM_STATES * 4] = 0.f;
	eigenvectorsInverseWrtPrimitives[2 + NUM_STATES * 5] = -alphaFast * betaY / (2.f * sqrtDensity * speedOfSound);
	eigenvectorsInverseWrtPrimitives[2 + NUM_STATES * 6] = -alphaFast * betaZ / (2.f * sqrtDensity * speedOfSound);
	eigenvectorsInverseWrtPrimitives[2 + NUM_STATES * 7] = alphaSlow / (2.f * density * speedOfSoundSq);
	//entropy row
	eigenvectorsInverseWrtPrimitives[3 + NUM_STATES * 0] = 1.f;
	eigenvectorsInverseWrtPrimitives[3 + NUM_STATES * 1] = 0.f;
	eigenvectorsInverseWrtPrimitives[3 + NUM_STATES * 2] = 0.f;
	eigenvectorsInverseWrtPrimitives[3 + NUM_STATES * 3] = 0.f;
	eigenvectorsInverseWrtPrimitives[3 + NUM_STATES * 4] = 0.f;
	eigenvectorsInverseWrtPrimitives[3 + NUM_STATES * 5] = 0.f;
	eigenvectorsInverseWrtPrimitives[3 + NUM_STATES * 6] = 0.f;
	eigenvectorsInverseWrtPrimitives[3 + NUM_STATES * 7] = -1.f / speedOfSoundSq;
	//divergence row
	eigenvectorsInverseWrtPrimitives[4 + NUM_STATES * 0] = 0.f;
	eigenvectorsInverseWrtPrimitives[4 + NUM_STATES * 1] = 0.f;
	eigenvectorsInverseWrtPrimitives[4 + NUM_STATES * 2] = 0.f;
	eigenvectorsInverseWrtPrimitives[4 + NUM_STATES * 3] = 0.f;
	eigenvectorsInverseWrtPrimitives[4 + NUM_STATES * 4] = 1.f;
	eigenvectorsInverseWrtPrimitives[4 + NUM_STATES * 5] = 0.f;
	eigenvectorsInverseWrtPrimitives[4 + NUM_STATES * 6] = 0.f;
	eigenvectorsInverseWrtPrimitives[4 + NUM_STATES * 7] = 0.f;
	//slow magnetoacoustic row
	eigenvectorsInverseWrtPrimitives[5 + NUM_STATES * 0] = 0.f;
	eigenvectorsInverseWrtPrimitives[5 + NUM_STATES * 1] = alphaSlow * slowSpeed * .5f / speedOfSoundSq;
	eigenvectorsInverseWrtPrimitives[5 + NUM_STATES * 2] = alphaFast * fastSpeed * betaY * sgnBx * .5f / speedOfSoundSq;
	eigenvectorsInverseWrtPrimitives[5 + NUM_STATES * 3] = alphaFast * fastSpeed * betaZ * sgnBx * .5f / speedOfSoundSq;
	eigenvectorsInverseWrtPrimitives[5 + NUM_STATES * 4] = 0.f;
	eigenvectorsInverseWrtPrimitives[5 + NUM_STATES * 5] = -alphaFast * betaY / (2.f * sqrtDensity * speedOfSound);
	eigenvectorsInverseWrtPrimitives[5 + NUM_STATES * 6] = -alphaFast * betaZ / (2.f * sqrtDensity * speedOfSound);
	eigenvectorsInverseWrtPrimitives[5 + NUM_STATES * 7] = alphaSlow / (2.f * density * speedOfSoundSq);
	//Alfven row
	eigenvectorsInverseWrtPrimitives[6 + NUM_STATES * 0] = 0.f;
	eigenvectorsInverseWrtPrimitives[6 + NUM_STATES * 1] = 0.f;
	eigenvectorsInverseWrtPrimitives[6 + NUM_STATES * 2] = -betaZ / sqrtDensity;
	eigenvectorsInverseWrtPrimitives[6 + NUM_STATES * 3] = betaY / sqrtDensity;
	eigenvectorsInverseWrtPrimitives[6 + NUM_STATES * 4] = 0.f;
	eigenvectorsInverseWrtPrimitives[6 + NUM_STATES * 5] = betaZ * M_SQRT_1_2 / sqrtDensity;
	eigenvectorsInverseWrtPrimitives[6 + NUM_STATES * 6] = -betaY * M_SQRT_1_2 / sqrtDensity;
	eigenvectorsInverseWrtPrimitives[6 + NUM_STATES * 7] = 0.f;
	//fast magnetoacoustic row
	eigenvectorsInverseWrtPrimitives[7 + NUM_STATES * 0] = 0.f;
	eigenvectorsInverseWrtPrimitives[7 + NUM_STATES * 1] = alphaFast * fastSpeed / (2.f * speedOfSoundSq);
	eigenvectorsInverseWrtPrimitives[7 + NUM_STATES * 2] = -alphaSlow * slowSpeed * betaY * sgnBx / (2.f * speedOfSoundSq);
	eigenvectorsInverseWrtPrimitives[7 + NUM_STATES * 3] = -alphaSlow * slowSpeed * betaZ * sgnBx / (2.f * speedOfSoundSq);
	eigenvectorsInverseWrtPrimitives[7 + NUM_STATES * 4] = 0.f;
	eigenvectorsInverseWrtPrimitives[7 + NUM_STATES * 5] = alphaSlow * betaY / (2.f * sqrtDensity * speedOfSound);
	eigenvectorsInverseWrtPrimitives[7 + NUM_STATES * 6] = alphaSlow * betaZ / (2.f * sqrtDensity * speedOfSound);
	eigenvectorsInverseWrtPrimitives[7 + NUM_STATES * 7] = alphaFast / (2.f * density * speedOfSoundSq);

	//left and right eigenvectors above are of the flux derivative with respect to primitive variables
	//to find the eigenvectors of the flux with respect to the state variables, multiply by the derivative of the primitives with respect to the states
	//L = l * dw/du, R = du/dw * r
	//for l, r the left and right eigenvectors of derivative of flux wrt primitives
	//u = states, w = primitives
	//L, R the left and right eigenvectors of derivative of flux wrt state
	//this matches up with A = Q V Q^-1 = R V L = du/dw r V l dw/du

	real8 du_dw8[8];	//row-major
	du_dw8[0] = (real8)(1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
	du_dw8[1] = (real8)(velocity.x, density, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
	du_dw8[2] = (real8)(velocity.y, 0.f, density, 0.f, 0.f, 0.f, 0.f, 0.f);
	du_dw8[3] = (real8)(velocity.z, 0.f, 0.f, density, 0.f, 0.f, 0.f, 0.f);
	du_dw8[4] = (real8)(0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f);
	du_dw8[5] = (real8)(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f);
	du_dw8[6] = (real8)(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f);
	du_dw8[7] = (real8)(.5f * velocitySq, density * velocity.x, density * velocity.y, density * velocity.z, magneticField.x, magneticField.y, magneticField.z, 1.f / gammaMinusOne);
	real* du_dw = (real*)du_dw8;

	real8 dw_du8[8];	//row-major
	dw_du8[0] = (real8)(1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
	dw_du8[1] = (real8)(-velocity.x / density, 1.f / density, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
	dw_du8[2] = (real8)(-velocity.y / density, 0.f, 1.f / density, 0.f, 0.f, 0.f, 0.f, 0.f);
	dw_du8[3] = (real8)(-velocity.z / density, 0.f, 0.f, 1.f / density, 0.f, 0.f, 0.f, 0.f);
	dw_du8[4] = (real8)(0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f);
	dw_du8[5] = (real8)(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f);
	dw_du8[6] = (real8)(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f);
	dw_du8[7] = (real8)(.5f * gammaMinusOne * velocitySq, -gammaMinusOne * velocity.x, -gammaMinusOne * velocity.y, -gammaMinusOne * velocity.z, -gammaMinusOne * magneticField.x, -gammaMinusOne * magneticField.y, -gammaMinusOne * magneticField.z, gammaMinusOne);
	real* dw_du = (real*)dw_du8;

	//L = l * dw/du <=> L_j = l_k * [dw/du]_kj <=> L_ij = l_ik * [dw/du]_kj
	//R = du/dw * r <=> R_i = [du/dw]_ik * r_k <=> R_ij = [du/dw]_ik * r_kj
	for (int i = 0; i < NUM_STATES; ++i) {
		for (int j = 0; j < NUM_STATES; ++j) {
			real sum;
			
			sum = 0.f;
			for (int k = 0; k < NUM_STATES; ++k) {
				sum += eigenvectorsInverseWrtPrimitives[i + NUM_STATES * k] * dw_du[k + NUM_STATES * j];
			}
			eigenvectorsInverse[i + NUM_STATES * j] = sum;
			
			sum = 0.f;
			for (int k = 0; k < NUM_STATES; ++k) {
				sum += du_dw[i + NUM_STATES * k] * eigenvectorsWrtPrimitives[k + NUM_STATES * j];
			}
			eigenvectors[i + NUM_STATES * j] = sum;
		}
	}

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


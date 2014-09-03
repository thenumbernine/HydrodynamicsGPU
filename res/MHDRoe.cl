/*
using the following:
"A Numerical Solution of Hyperbolic Partial Differential Equations", Trangenstein, 2007
"A multidimensional upwind scheme for magnetohydrodynamics" by Falle, Komissarov, Joarder, 1998
"A Solution-Adaptive Upwind Scheme for Ideal Magnetohydrodynamics" by Powell, Roe, Linde, Gombosi, Zeeuw, 1999

The eigenvalues of all three papers match up (with the exception of Powel 1999, which mentions Alfven waves being |B|/sqrt(rho) then defines the Alfven wave eigenvalue to be B/rho, no sqrt, no abs, but I'm pretty sure that's a typo)

The Alfven wave eigenvectors of the 199x papers and Trangenstein match up (with the exception of normalization, which Trangenstein neglects because he ignores computing the left-eigenvectors, which could absorb the normalization term) 

Now on to the fast and slow wave eigenvectors ... 


And the degeneracy of Bx=0:

Powell 1999 says something along the lines of "the system simply reduces to the Euler equation eigenvectors". By this they mean you'll have to write a separate case for Bx=0 with eigenvectors similar to those of the Euler equations'.
A subtle mention of what to do with those extra transverse magetic field values: add them to the pressure.
No mention that I noticed of how the system, on its own, will not reduce to the Euler equations, and in the absense of correctly calculated limits of ratios approaching zero, you will end up with the speed of sound squared in a few places where you should've got a zero.
The separate case for Bx=0 is necessary.

Trangenstein gives an exact example of the rhs eigenvectors in the Bx=0 case.  Thank you.
If only he gave left eigenvectors too.  Good thing most of them match with Powell and Falle.


Powell 1999 also doesn't state that "a" is the speed of sound.  Browsing through their sources clarifies the "a". 
*/

#include "HydroGPU/Shared/Common.h"

#define DONT_EVOLVE_MAGNETIC_FIELD

#if NUM_STATES != 8
#error MHD expects 8 state variables
#endif

void invert8x8(real* dst, const real* src);
void invert8x8(real* dst, const real* src) {
	real tmp[8*8];
	for (int i = 0; i < 64; ++i) {
		tmp[i] = src[i];
	}
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			dst[i + 8 * j] = i == j ? 1.f : 0.f;
		}
	}

	const float epsilon = 1e-7f;

	for (int j = 0; j < 8; ++j) {   //jth col/row
		//make sure diagonal is 1
		if (fabs(tmp[j + 8 * j] - 1.f) > epsilon) {
			real s = 1.f / tmp[j + 8 * j];
			for (int k = 0; k < 8; ++k) {
				tmp[j + 8 * k] *= s;
			}
			for (int k = 0; k < 8; ++k) {
				dst[j + 8 * k] *= s;
			}
		}

		for (int i = 0; i < 8; ++i) {   //ith row - get rid of lower elements
			if (i == j) continue;
			real a = tmp[i + 8 * j];
			if (fabs(a) > epsilon) { //nonzero entry ...
				for (int k = 0; k < 8; ++k) {
					tmp[i + 8 * k] -= a * tmp[j + 8 * k];   //tmp(j,j) should already be 1 by the loop above
				}
				for (int k = 0; k < 8; ++k) {
					dst[i + 8 * k] -= a * dst[j + 8 * k];
				}
			}
		}
	}
}

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
	real roeWeightL = sqrt(densityL);

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
	real roeWeightR = sqrt(densityR);

	//3.5.2 "In this paper, a simple arithmetic averaging of the primitive variables is done to compute the interface state."
	//but while I'm solving the degeneracy case, I'll use Roe weighting
	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real4 velocity = (velocityL * roeWeightL + velocityR * roeWeightR) * roeWeightNormalization;
	real4 magneticField = (magneticFieldL * roeWeightL + magneticFieldR * roeWeightR) * roeWeightNormalization;
	real pressure = (pressureL * roeWeightL + pressureR * roeWeightR) * roeWeightNormalization;
	real enthalpyTotal = (enthalpyTotalL * roeWeightL + enthalpyTotalR * roeWeightR) * roeWeightNormalization;
	real potentialEnergy = (potentialEnergyL * roeWeightL + potentialEnergyR * roeWeightR) * roeWeightNormalization;
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
	
	real magneticFieldSq = dot(magneticField, magneticField);
	real magneticFieldXSq = magneticField.x * magneticField.x;
	
	real magneticFieldTSq = magneticField.y * magneticField.y + magneticField.z * magneticField.z;
	real magneticFieldT = sqrt(magneticFieldTSq);

#define M_SQRT_1_2	0.7071067811865475727373109293694142252206802368164f

#define DEBUG_INDEX		-1	//512

	//matrices are stored as A_ij = A[i + height * j]
#ifndef DONT_EVOLVE_MAGNETIC_FIELD
	if (fabs(magneticField.x) < 1e-7f && magneticFieldT < 1e-7f) {	//magnetic field is empty 
#endif	
if (index == DEBUG_INDEX) {
printf("magnetic field n=0, t=0\n");
}	

		real speedOfSoundSq = (enthalpyTotal - .5f * velocitySq - potentialEnergy) * (gamma - 1.f);
		real speedOfSound = sqrt(speedOfSoundSq);

		eigenvalues[0] = velocity.x - speedOfSound;
		eigenvalues[1] = velocity.x;
		eigenvalues[2] = velocity.x;
		eigenvalues[3] = velocity.x;
		eigenvalues[4] = 0.f;
		eigenvalues[5] = 0.f;
		eigenvalues[6] = 0.f;
		eigenvalues[7] = velocity.x + speedOfSound;

		//slow
		eigenvectors[0 + NUM_STATES * 0] = 1.f;
		eigenvectors[1 + NUM_STATES * 0] = velocity.x - speedOfSound;
		eigenvectors[2 + NUM_STATES * 0] = velocity.y;
		eigenvectors[3 + NUM_STATES * 0] = velocity.z;
		eigenvectors[4 + NUM_STATES * 0] = 0.f;
		eigenvectors[5 + NUM_STATES * 0] = 0.f;
		eigenvectors[6 + NUM_STATES * 0] = 0.f;
		eigenvectors[7 + NUM_STATES * 0] = enthalpyTotal - speedOfSound * velocity.x;
		//mid vel
		eigenvectors[0 + NUM_STATES * 1] = 1.f;
		eigenvectors[1 + NUM_STATES * 1] = velocity.x;
		eigenvectors[2 + NUM_STATES * 1] = velocity.y;
		eigenvectors[3 + NUM_STATES * 1] = velocity.z;
		eigenvectors[4 + NUM_STATES * 1] = 0.f;
		eigenvectors[5 + NUM_STATES * 1] = 0.f;
		eigenvectors[6 + NUM_STATES * 1] = 0.f;
		eigenvectors[7 + NUM_STATES * 1] = .5f * velocitySq;
		//mid vel
		eigenvectors[0 + NUM_STATES * 2] = 0.f;
		eigenvectors[1 + NUM_STATES * 2] = 0.f;
		eigenvectors[2 + NUM_STATES * 2] = 1.f;
		eigenvectors[3 + NUM_STATES * 2] = 0.f;
		eigenvectors[4 + NUM_STATES * 2] = 0.f;
		eigenvectors[5 + NUM_STATES * 2] = 0.f;
		eigenvectors[6 + NUM_STATES * 2] = 0.f;
		eigenvectors[7 + NUM_STATES * 2] = velocity.y;
		//mid vel
		eigenvectors[0 + NUM_STATES * 3] = 0.f;
		eigenvectors[1 + NUM_STATES * 3] = 0.f;
		eigenvectors[2 + NUM_STATES * 3] = 0.f;
		eigenvectors[3 + NUM_STATES * 3] = 1.f;
		eigenvectors[4 + NUM_STATES * 3] = 0.f;
		eigenvectors[5 + NUM_STATES * 3] = 0.f;
		eigenvectors[6 + NUM_STATES * 3] = 0.f;
		eigenvectors[7 + NUM_STATES * 3] = velocity.z;
		//mid magnetic
		eigenvectors[0 + NUM_STATES * 4] = 0.f;
		eigenvectors[1 + NUM_STATES * 4] = 0.f;
		eigenvectors[2 + NUM_STATES * 4] = 0.f;
		eigenvectors[3 + NUM_STATES * 4] = 0.f;
		eigenvectors[4 + NUM_STATES * 4] = 1.f;
		eigenvectors[5 + NUM_STATES * 4] = 0.f;
		eigenvectors[6 + NUM_STATES * 4] = 0.f;
		eigenvectors[7 + NUM_STATES * 4] = 0.f;
		//mid magnetic
		eigenvectors[0 + NUM_STATES * 5] = 0.f;
		eigenvectors[1 + NUM_STATES * 5] = 0.f;
		eigenvectors[2 + NUM_STATES * 5] = 0.f;
		eigenvectors[3 + NUM_STATES * 5] = 0.f;
		eigenvectors[4 + NUM_STATES * 5] = 0.f;
		eigenvectors[5 + NUM_STATES * 5] = 1.f;
		eigenvectors[6 + NUM_STATES * 5] = 0.f;
		eigenvectors[7 + NUM_STATES * 5] = 0.f;
		//mid magnetic
		eigenvectors[0 + NUM_STATES * 6] = 0.f;
		eigenvectors[1 + NUM_STATES * 6] = 0.f;
		eigenvectors[2 + NUM_STATES * 6] = 0.f;
		eigenvectors[3 + NUM_STATES * 6] = 0.f;
		eigenvectors[4 + NUM_STATES * 6] = 0.f;
		eigenvectors[5 + NUM_STATES * 6] = 0.f;
		eigenvectors[6 + NUM_STATES * 6] = 1.f;
		eigenvectors[7 + NUM_STATES * 6] = 0.f;
		//fast
		eigenvectors[0 + NUM_STATES * 7] = 1.f;
		eigenvectors[1 + NUM_STATES * 7] = velocity.x + speedOfSound;
		eigenvectors[2 + NUM_STATES * 7] = velocity.y;
		eigenvectors[3 + NUM_STATES * 7] = velocity.z;
		eigenvectors[4 + NUM_STATES * 7] = 0.f;
		eigenvectors[5 + NUM_STATES * 7] = 0.f;
		eigenvectors[6 + NUM_STATES * 7] = 0.f;
		eigenvectors[7 + NUM_STATES * 7] = enthalpyTotal + speedOfSound * velocity.x;

		real invDenom = .5f / (speedOfSound * speedOfSound);
		//min row
		eigenvectorsInverse[0 + NUM_STATES * 0] = (.5f * (gamma - 1.f) * velocitySq + speedOfSound * velocity.x) * invDenom;
		eigenvectorsInverse[0 + NUM_STATES * 1] = -(speedOfSound + (gamma - 1.f) * velocity.x) * invDenom;
		eigenvectorsInverse[0 + NUM_STATES * 2] = -(gamma - 1.f) * velocity.y * invDenom;
		eigenvectorsInverse[0 + NUM_STATES * 3] = -(gamma - 1.f) * velocity.z * invDenom;
		eigenvectorsInverse[0 + NUM_STATES * 4] = 0.f;
		eigenvectorsInverse[0 + NUM_STATES * 5] = 0.f;
		eigenvectorsInverse[0 + NUM_STATES * 6] = 0.f;
		eigenvectorsInverse[0 + NUM_STATES * 7] = (gamma - 1.f) * invDenom;
		//mid normal row
		eigenvectorsInverse[1 + NUM_STATES * 0] = 1.f - (gamma - 1.f) * velocitySq * invDenom;
		eigenvectorsInverse[1 + NUM_STATES * 1] = (gamma - 1.f) * velocity.x * 2.f * invDenom;
		eigenvectorsInverse[1 + NUM_STATES * 2] = (gamma - 1.f) * velocity.y * 2.f * invDenom;
		eigenvectorsInverse[1 + NUM_STATES * 3] = (gamma - 1.f) * velocity.z * 2.f * invDenom;
		eigenvectorsInverse[1 + NUM_STATES * 4] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 5] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 6] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 7] = -(gamma - 1.f) * 2.f * invDenom;
		//mid tangent A row
		eigenvectorsInverse[2 + NUM_STATES * 0] = -velocity.y; 
		eigenvectorsInverse[2 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[2 + NUM_STATES * 2] = 1.f;
		eigenvectorsInverse[2 + NUM_STATES * 3] = 0.f;
		eigenvectorsInverse[2 + NUM_STATES * 4] = 0.f;
		eigenvectorsInverse[2 + NUM_STATES * 5] = 0.f;
		eigenvectorsInverse[2 + NUM_STATES * 6] = 0.f;
		eigenvectorsInverse[2 + NUM_STATES * 7] = 0.f;
		//mid tangent B row
		eigenvectorsInverse[3 + NUM_STATES * 0] = -velocity.z;
		eigenvectorsInverse[3 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 2] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 3] = 1.f;
		eigenvectorsInverse[3 + NUM_STATES * 4] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 5] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 6] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 7] = 0.f;
		//magnetic row
		eigenvectorsInverse[4 + NUM_STATES * 0] = 0.f;
		eigenvectorsInverse[4 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[4 + NUM_STATES * 2] = 0.f;
		eigenvectorsInverse[4 + NUM_STATES * 3] = 0.f;
		eigenvectorsInverse[4 + NUM_STATES * 4] = 1.f;
		eigenvectorsInverse[4 + NUM_STATES * 5] = 0.f;
		eigenvectorsInverse[4 + NUM_STATES * 6] = 0.f;
		eigenvectorsInverse[4 + NUM_STATES * 7] = 0.f;
		//magnetic row
		eigenvectorsInverse[5 + NUM_STATES * 0] = 0.f;
		eigenvectorsInverse[5 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[5 + NUM_STATES * 2] = 0.f;
		eigenvectorsInverse[5 + NUM_STATES * 3] = 0.f;
		eigenvectorsInverse[5 + NUM_STATES * 4] = 0.f;
		eigenvectorsInverse[5 + NUM_STATES * 5] = 1.f;
		eigenvectorsInverse[5 + NUM_STATES * 6] = 0.f;
		eigenvectorsInverse[5 + NUM_STATES * 7] = 0.f;
		//magnetic row
		eigenvectorsInverse[6 + NUM_STATES * 0] = 0.f;
		eigenvectorsInverse[6 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[6 + NUM_STATES * 2] = 0.f;
		eigenvectorsInverse[6 + NUM_STATES * 3] = 0.f;
		eigenvectorsInverse[6 + NUM_STATES * 4] = 0.f;
		eigenvectorsInverse[6 + NUM_STATES * 5] = 0.f;
		eigenvectorsInverse[6 + NUM_STATES * 6] = 1.f;
		eigenvectorsInverse[6 + NUM_STATES * 7] = 0.f;
		//max row
		eigenvectorsInverse[7 + NUM_STATES * 0] = (.5f * (gamma - 1.f) * velocitySq - speedOfSound * velocity.x) * invDenom;
		eigenvectorsInverse[7 + NUM_STATES * 1] = (speedOfSound - (gamma - 1.f) * velocity.x) * invDenom;
		eigenvectorsInverse[7 + NUM_STATES * 2] = -(gamma - 1.f) * velocity.y * invDenom;
		eigenvectorsInverse[7 + NUM_STATES * 3] = -(gamma - 1.f) * velocity.z * invDenom;
		eigenvectorsInverse[7 + NUM_STATES * 4] = 0.f;
		eigenvectorsInverse[7 + NUM_STATES * 5] = 0.f;
		eigenvectorsInverse[7 + NUM_STATES * 6] = 0.f;
		eigenvectorsInverse[7 + NUM_STATES * 7] = (gamma - 1.f) * invDenom;
#ifndef DONT_EVOLVE_MAGNETIC_FIELD
	} else {
	
		real speedOfSound = sqrt(gamma * pressure / density);
		real speedOfSoundSq = speedOfSound * speedOfSound;

		real sqrtDensity = sqrt(density);
		real AlfvenSpeed = fabs(magneticField.x) / sqrtDensity;
		real starSpeedSq = .5f * (speedOfSoundSq + magneticFieldSq / density);
		real discr = starSpeedSq * starSpeedSq - speedOfSoundSq * magneticFieldXSq / density;
		real tmp = sqrt(discr);
		real fastSpeedSq = starSpeedSq + tmp;
		real fastSpeed = sqrt(fastSpeedSq);
		real slowSpeedSq = starSpeedSq - tmp;
		real slowSpeed = sqrt(slowSpeedSq);
		
		real sgnBx;
		if (magneticField.x > 0.f) {
			sgnBx = 1.f;
		} else {
			sgnBx = -1.f;
		}

		//eigenvalues

		eigenvalues[0] = velocity.x - fastSpeed;
		eigenvalues[1] = velocity.x - AlfvenSpeed;
		eigenvalues[2] = velocity.x - slowSpeed;
		eigenvalues[3] = velocity.x;
		eigenvalues[4] = velocity.x;
		eigenvalues[5] = velocity.x + slowSpeed;
		eigenvalues[6] = velocity.x + AlfvenSpeed;
		eigenvalues[7] = velocity.x + fastSpeed;

		//eigenvectors
		real eigenvectorsWrtPrimitives[NUM_STATES * NUM_STATES];
		
		//eigenvectors inverse
		real eigenvectorsInverseWrtPrimitives[NUM_STATES * NUM_STATES];

		if (fabs(magneticField.x) < 1e-7f) {	//magnetic field has no normal component

if (index == DEBUG_INDEX) {
printf("magnetic field n=0\n");
}

			//alfven speed is zero, slow speed is zero
			//fast speed is almost the speed of sound ... with the tangent magnetism mixed in there

/*
cf^2 = c*^2 + sqrt(c*^4 - c^2 Bx^2 / rho)
c*^2 = 1/2 (c^2 + B^2 / rho)
r_11 = (rho * cf^2 - B^2) / c^2

when Bx = 0
cf^2 = c*^2 + sqrt(c*^4)
cf^2 = c*^2 + c*^2
cf^2 = 2 * c*^2

c*^2 = 1/2 (c^2 + (By^2 + Bz^2) / rho)
cf^2 = c^2 + (By^2 + Bz^2) / rho

r_11 = (rho * cf^2 - By^2 - Bz^2) / c^2
r_11 = (rho * (c^2 + (By^2 + Bz^2) / rho) - By^2 - Bz^2) / c^2
r_11 = (rho * c^2 + By^2 + Bz^2 - By^2 - Bz^2) / c^2
r_11 = rho * c^2 / c^2
r_11 = rho
*/

			//fast magnetoacoustic col 
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 0] = (density * fastSpeedSq - magneticFieldTSq) / speedOfSoundSq;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 0] = -fastSpeed;
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 0] = 0.f;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 0] = 0.f;
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 0] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 0] = magneticField.y;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 0] = magneticField.z;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 0] = density * fastSpeedSq - magneticFieldTSq;
			//Alfven col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 1] = 1.f;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 1] = 0.f;
			//slow magnetoacoustic col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 2] = 0.f;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 2] = 0.f;
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 2] = 1.f;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 2] = 0.f;
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 2] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 2] = 0.f;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 2] = 0.f;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 2] = 0.f;
			//entropy col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 3] = 0.f;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 3] = 0.f; 
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 3] = 0.f;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 3] = 1.f;
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
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 5] = 0.f;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 5] = 0.f;
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 5] = 0.f;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 5] = 0.f;
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 5] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 5] = 1.f;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 5] = 0.f;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 5] = -magneticField.y;
			//Alfven col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 6] = 1.f;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 6] = -magneticField.z;
			//fast magnetoacoustic col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 7] = (density * fastSpeedSq - magneticFieldTSq) / speedOfSoundSq;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 7] = fastSpeed;
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 7] = 0.f;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 7] = 0.f;
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 7] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 7] = magneticField.y;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 7] = magneticField.z;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 7] = density * fastSpeedSq - magneticFieldTSq;

		} else {
			
if (index == DEBUG_INDEX) {
printf("magnetic field in n and t\n");
}
			//fast magnetoacoustic col 
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 0] = (density * fastSpeedSq - magneticFieldSq) / speedOfSoundSq;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 0] = -fastSpeed + magneticFieldXSq / (density * fastSpeed);
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 0] = magneticField.x * magneticField.y / (density * fastSpeed);
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 0] = magneticField.x * magneticField.z / (density * fastSpeed);
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 0] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 0] = magneticField.y;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 0] = magneticField.z;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 0] = density * fastSpeedSq - magneticFieldSq;
			//Alfven col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 1] = sgnBx * magneticField.z;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 1] = -sgnBx * magneticField.y;
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 1] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 1] = magneticField.z * sqrtDensity;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 1] = -magneticField.y * sqrtDensity;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 1] = 0.f;
			//slow magnetoacoustic col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 2] = (density * slowSpeedSq - magneticFieldSq) / speedOfSoundSq;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 2] = -slowSpeed + magneticFieldXSq / (density * slowSpeed);
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 2] = magneticField.x * magneticField.y / (density * slowSpeed);
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 2] = magneticField.x * magneticField.z / (density * slowSpeed);
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 2] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 2] = magneticField.y;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 2] = magneticField.z;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 2] = density * slowSpeedSq - magneticFieldSq;
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
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 5] = (density * slowSpeedSq - magneticFieldSq) / speedOfSoundSq;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 5] = slowSpeed - magneticFieldXSq / (density * slowSpeed);
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 5] = -magneticField.x * magneticField.y / (density * slowSpeed);
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 5] = -magneticField.x * magneticField.z / (density * slowSpeed);
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 5] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 5] = magneticField.y;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 5] = magneticField.z;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 5] = density * slowSpeedSq - magneticFieldSq;
			//Alfven col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 6] = -sgnBx * magneticField.z;
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 6] = sgnBx * magneticField.y;
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 6] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 6] = magneticField.z * sqrtDensity;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 6] = -magneticField.y * sqrtDensity;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 6] = 0.f;
			//fast magnetoacoustic col
			eigenvectorsWrtPrimitives[0 + NUM_STATES * 7] = (density * fastSpeedSq - magneticFieldSq) / speedOfSoundSq;
			eigenvectorsWrtPrimitives[1 + NUM_STATES * 7] = fastSpeed - magneticFieldXSq / (density * fastSpeed);
			eigenvectorsWrtPrimitives[2 + NUM_STATES * 7] = -magneticField.x * magneticField.y / (density * fastSpeed);
			eigenvectorsWrtPrimitives[3 + NUM_STATES * 7] = -magneticField.x * magneticField.z / (density * fastSpeed);
			eigenvectorsWrtPrimitives[4 + NUM_STATES * 7] = 0.f;
			eigenvectorsWrtPrimitives[5 + NUM_STATES * 7] = magneticField.y;
			eigenvectorsWrtPrimitives[6 + NUM_STATES * 7] = magneticField.z;
			eigenvectorsWrtPrimitives[7 + NUM_STATES * 7] = density * fastSpeedSq - magneticFieldSq;
		}

		invert8x8(eigenvectorsInverseWrtPrimitives, eigenvectorsWrtPrimitives);

		//for all but the no-magnetic-field case transform the eigenvectors by dw/du

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

#if 1
		if (index == DEBUG_INDEX) {
			printf("side %d\n", side);
			printf("i %d\n", index);
			//heart of current problem: magnetic energy density is exceeding our total energy density
			// so the K+P energy density comes out negative ...
			//magnetic energy density comes from the magnetic field states
			//total energy density comes from the the ENERGY_TOTAL state
			// this means our eigenvectors are contributing less to total energy than they should be. 
			printf("totalPlasmaEnergyDensityL %f\n", totalPlasmaEnergyDensityL);
			printf("magneticEnergyDensityL %f\n", magneticEnergyDensityL);
			printf("potentialEnergyDensityL %f\n", potentialEnergyDensityL);
			printf("kineticEnergyDensityL %f\n", kineticEnergyDensityL);
			printf("totalHydroEnergyDensityL %f\n", totalHydroEnergyDensityL);	//shouldn't be negative
			printf("internalEnergyDensityL %f\n", internalEnergyDensityL);
			printf("density %f\n", density);
			printf("gamma %f\n", gamma);
			printf("pressureL %f\n", pressureL);
			printf("pressureR %f\n", pressureR);
			printf("pressure %f\n", pressure);
			printf("speedOfSound %f\n", speedOfSound);
			//printf("slowSpeed %f\n", slowSpeed);
			printf("magnetic field %f %f %f\n", magneticField.x, magneticField.y, magneticField.z);
			printf("eigenvalues");
			for (int i = 0; i < NUM_STATES; ++i) {
				printf(" %f", eigenvalues[i]);
			}
			printf("\n");
			printf("primitive eigenvectors\n");
			for (int i = 0; i < NUM_STATES; ++i) {
				for (int j = 0; j < NUM_STATES; ++j) {
					printf(" %f", eigenvectorsWrtPrimitives[i + NUM_STATES * j]);
				}
				printf("\n");
			}
			printf("primitive eigenvector inverse\n");
			for (int i = 0; i < NUM_STATES; ++i) {
				for (int j = 0; j < NUM_STATES; ++j) {
					printf(" %f", eigenvectorsInverseWrtPrimitives[i + NUM_STATES * j]);
				}
				printf("\n");
			}	
			printf("primitive eigenbasis orthogonality\n");
			real totalError = 0.f;
			for (int i = 0; i < NUM_STATES; ++i) {
				for (int j = 0; j < NUM_STATES; ++j) {
					real sum = 0.f;
					for (int k = 0; k < NUM_STATES; ++k) {
						sum += eigenvectorsInverseWrtPrimitives[i + NUM_STATES * k] * eigenvectorsWrtPrimitives[k + NUM_STATES * j];
					}
					printf(" %f", sum);
					totalError += fabs(sum - (i == j ? 1.f : 0.f));
				}
				printf("\n");
			}
			printf("eigenbasis error %f\n", totalError);
		}
#endif

	
	}


#if 1
	if (index == DEBUG_INDEX) {
		printf("side %d\n", side);
		printf("i %d\n", index);
		printf("conservative eigenbasis orthogonality\n");
		real totalError = 0.f;
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
		printf("eigenbasis error %f\n", totalError);
	}
#endif
#endif	//DONT_EVOLVE_MAGNETIC_FIELD

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


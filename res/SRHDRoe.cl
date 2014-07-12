/*
The components of the Roe solver specific to the Euler equations
paritcularly the spectral decomposition
*/

#include "HydroGPU/Shared/Common.h"

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* primitiveBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	int side);

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* primitiveBuffer,
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
	
	const __global real* primitiveL = primitiveBuffer + NUM_PRIMITIVE * indexPrev;
	const __global real* primitiveR = primitiveBuffer + NUM_PRIMITIVE * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
	__global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;

	real properRestMassDensityL = primitiveL[PRIMITIVE_DENSITY];	//rho
	real restMassDensityL = stateL[STATE_REST_MASS_DENSITY];	//D
	real lorentzFactorL = restMassDensityL / properRestMassDensityL;	//W = u0
	real lorentzFactorSqL = lorentzFactorL * lorentzFactorL;
	real4 newtonianVelocityL = (real4)(0.f, 0.f, 0.f, 0.f);
	newtonianVelocityL.x = primitiveL[PRIMITIVE_VELOCITY_X];	//vi
#if DIM > 1
	newtonianVelocityL.y = primitiveL[PRIMITIVE_VELOCITY_Y];
#if DIM > 2
	newtonianVelocityL.z = primitiveL[PRIMITIVE_VELOCITY_Z];
#endif
#endif
	real pressureL = primitiveL[PRIMITIVE_PRESSURE];					//rho
	real4 relativisticVelocityL = newtonianVelocityL * lorentzFactorL;	//ui = vi * u0 = vi * W
	real totalEnergyDensityL = stateL[STATE_TOTAL_ENERGY_DENSITY];		//tau
	real internalSpecificEnthalpyL = (totalEnergyDensityL + pressureL + restMassDensityL) / (properRestMassDensityL * lorentzFactorSqL);	//h
	real pressureOverProperDensityEnthalpyL = pressureL / (properRestMassDensityL * internalSpecificEnthalpyL);	//P / (rho h)
	const real sqrtMetricL = 1.f;
	real roeWeightL = sqrt(sqrtMetricL * properRestMassDensityL * internalSpecificEnthalpyL);

	real properRestMassDensityR = primitiveR[PRIMITIVE_DENSITY];	//rho
	real restMassDensityR = stateR[STATE_REST_MASS_DENSITY];	//D
	real lorentzFactorR = restMassDensityR / properRestMassDensityR;	//W = u0
	real lorentzFactorSqR = lorentzFactorR * lorentzFactorR;
	real4 newtonianVelocityR = (real4)(0.f, 0.f, 0.f, 0.f);
	newtonianVelocityR.x = primitiveR[PRIMITIVE_VELOCITY_X];	//vi
#if DIM > 1
	newtonianVelocityR.y = primitiveR[PRIMITIVE_VELOCITY_Y];
#if DIM > 2
	newtonianVelocityR.z = primitiveR[PRIMITIVE_VELOCITY_Z];
#endif
#endif
	real pressureR = primitiveR[PRIMITIVE_PRESSURE];
	real4 relativisticVelocityR = newtonianVelocityR * lorentzFactorR;	//ui = vi * u0
	real totalEnergyDensityR = stateR[STATE_TOTAL_ENERGY_DENSITY];
	real internalSpecificEnthalpyR = (totalEnergyDensityR + pressureR + restMassDensityR) / (properRestMassDensityR * lorentzFactorSqR);
	real pressureOverProperDensityEnthalpyR = pressureR / (properRestMassDensityR * internalSpecificEnthalpyR);
	const real sqrtMetricR = 1.f;
	real roeWeightR = sqrt(sqrtMetricR * properRestMassDensityR * internalSpecificEnthalpyR);

	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real lorentzFactor = (roeWeightL * lorentzFactorL + roeWeightR * lorentzFactorR) * roeWeightNormalization;	//W = u0
	real lorentzFactorSq = lorentzFactor * lorentzFactor;
	real4 relativisticVelocity = (roeWeightL * relativisticVelocityL + roeWeightR * relativisticVelocityR) * roeWeightNormalization;	//ui
	real pressureOverProperDensityEnthalpy = (roeWeightL * pressureOverProperDensityEnthalpyL + roeWeightR * pressureOverProperDensityEnthalpyR) * roeWeightNormalization;	//p / (rho h)
	real speedOfSoundSq = GAMMA * pressureOverProperDensityEnthalpy;
	real speedOfSound = sqrt(speedOfSoundSq);
	//how do we get 'h' from the Roe-weighted variables?

	//...now to calculate eigenvalues and vectors by the roe averaged variables	

	//calculate flux in x-axis and rotate into normal
	//works a bit more accurately than calculating by normal 

#if DIM > 1
	if (side == 1) {
		relativisticVelocity = (real4)(relativisticVelocity.y, -velocity.x, relativisticVelocity.z, 0.f);	// -90' rotation to put the y axis contents into the x axis
	} 
#if DIM > 2
	else if (side == 2) {
		relativisticVelocity = (real4)(relativisticVelocity.z, relativisticVelocity.y, -velocity.x, 0.f);	//-90' rotation to put the z axis in the x axis
	}
#endif
#endif

	real4 newtonianVelocity = relativisticVelocity / lorentzFactor;	//vi = ui / u0
	real newtonianVelocityXSq = newtonianVelocity.x * newtonianVelocity.x;
	real newtonianVelocitySq = dot(newtonianVelocity, newtonianVelocity);
	real denom = 1.f - newtonianVelocitySq * speedOfSoundSq;
	real a = newtonianVelocity.x * (1.f - speedOfSoundSq) / denom;
	real discr = (1.f - newtonianVelocitySq) * (1 - newtonianVelocityXSq - (newtonianVelocitySq - newtonianVelocityXSq) * speedOfSoundSq);
	real b = speedOfSound * sqrt(discr) / denom;
	//eigenvalues

	eigenvalues[0] = a - b;
	eigenvalues[1] = newtonianVelocity.x;
#if DIM > 1
	eigenvalues[2] = newtonianVelocity.x;
#if DIM > 2
	eigenvalues[3] = newtonianVelocity.x;
#endif
#endif
	eigenvalues[DIM+1] = a + b;

	//eigenvectors

	real Aminus = (1.f - newtonianVelocity.x * newtonianVelocity.x) / (1.f * newtonianVelocity.x * eigenvalues[0]);
	real Aplus = (1.f - newtonianVelocity.x * newtonianVelocity.x) / (1.f * newtonianVelocity.x * eigenvalues[DIM+1]);
	
	//TOOD how do you get h from P / (rho h) ?
	real K = internalSpecificEnthalpy;

	//min col 
	eigenvectors[0 + NUM_STATES * 0] = 1.f;
	eigenvectors[1 + NUM_STATES * 0] = internalSpecificEnthalpy * lorentzFactor * Aminus * eigenvalues[0];
#if DIM > 1
	eigenvectors[2 + NUM_STATES * 0] = internalSpecificEnthalpy * lorentzFactor * newtonianVelocity.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * 0] = internalSpecificEnthalpy * lorentzFactor * newtonianVelocity.z;
#endif
#endif
	eigenvectors[(DIM+1) + NUM_STATES * 0] = internalSpecificEnthalpy * lorentzFactor * Aminus - 1.f;
	//mid col (normal)
	eigenvectors[0 + NUM_STATES * 1] = K / (internalSpecificEnthalpy * lorentzFactor);
	eigenvectors[1 + NUM_STATES * 1] = newtonianVelocity.x;
#if DIM > 1
	eigenvectors[2 + NUM_STATES * 1] = newtonianVelocity.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * 1] = newtonianVelocity.z;
#endif
#endif
	eigenvectors[(DIM+1) + NUM_STATES * 1] = 1.f - K / (internalSpecificEnthalpy * lorentzFator);
	//mid col (tangent A)
#if DIM > 1
	eigenvectors[0 + NUM_STATES * 2] = lorentzFactor * newtonianVelocity.y;
	eigenvectors[1 + NUM_STATES * 2] = 2.f * internalSpecificEnthalpy * lorentzFactorSq * newtonianVelocity.x * newtonianVelocity.y;
	eigenvectors[2 + NUM_STATES * 2] = internalSpecificEnthalpy * (1.f + 2.f * lorentzFactorSq * newtonianVelocity.y * newtonianVelocity.y);
#if DIM > 2
	eigenvectors[3 + NUM_STATES * 2] = 2.f * internalSpecificEnthalpy * lorentzFatorSq * newtonianVelocity.y * newtonianVelocity.z;
#endif
	eigenvectors[(DIM+1) + NUM_STATES * 2] = (2.f * internalSpecificEnthalpy * lorentzFactorSq - lorentzFactor) * newtonianVelocity.y;
#endif
	//mid col (tangent B)
#if DIM > 2
	eigenvectors[0 + NUM_STATES * 3] = lorentzFactor * newtonianVelocity.z;
	eigenvectors[1 + NUM_STATES * 3] = 2.f * internalSpecificEnthalpy * lorentzFactorSq * newtonianVelocity.x * newtonianVelocity.z;
	eigenvectors[2 + NUM_STATES * 3] = 2.f * internalSpecificEnthalpy * lorentzFactorSq * newtonianVelocity.y * newtonianVelocity.z;
	eigenvectors[3 + NUM_STATES * 3] = internalSpecificEnthalpy * (1.f + 2.f * lorentzFactorSq * newtonianVelocity.z * newtonianVelocity.z);
	eigenvectors[(DIM+1) + NUM_STATES * 3] = (2.f * internalSpecificEnthalpy * lorentzFactorSq - lorentzFactor) * newtonianVelocity.z;
#endif
	//max col 
	eigenvectors[0 + NUM_STATES * (DIM+1)] = 1.f;
	eigenvectors[1 + NUM_STATES * (DIM+1)] = internalSpecificEnthalpy * lorentzFactor * Aplus * eigenvalues[DIM+1];
#if DIM > 1
	eigenvectors[2 + NUM_STATES * (DIM+1)] = internalSpecificEnthalpy * lorentzFactor * newtonianVelocity.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * (DIM+1)] = internalSpecificEnthalpy * lorentzFactor * newtonianVelocity.z;
#endif
#endif
	eigenvectors[(DIM+1) + NUM_STATES * (DIM+1)] = internalSpecificEnthalpy * lorentzFactor * Aplus - 1.f;

	
	//calculate eigenvector inverses ... 
	real determinant = internalSpecificEnthalpySq * internalSpecificEnthalpy * lorentzFactor * (K - 1.f) * (1.f - newtonianVelocityXSq) * (Aplus * eigenvalues[DIM+1] - Aminus * eigenvalues[0]);
	
	//min row
	real minMaxRowScale = internalSpecificEnthalpySq / determinant;
	eigenvectorsInverse[0 + NUM_STATES * 0] = 
		-minMaxRowScale * (internalSpecificEnthalpy * lorentzFactor * Aminus * (newtonianVelocity.x - eigenvalues[0]) 
		- newtonianVelocity.x 
		- lorentzFactorSq * (newtonianVelocitySq - newtonianVelocityXSq) * (2.f * K - 1.f) * (newtonianVelocity.x - Aminus * eigenvalues[0]) 
		+ K * Aminus * eigenvalues[0]);
	eigenvectorsInverse[0 + NUM_STATES * 1] = -minMaxRowScale * (1.f + lorentzFactorSq * (newtonianVelocitySq - newtonianVelocityXSq) * (2.f * K - 1.f) * (1.f - Aminus) - K * Aminus);
#if DIM > 1
	eigenvectorsInverse[0 + NUM_STATES * 2] = -minMaxRowScale * (lorentzFactorSq * newtonianVelocity.y * (2.f * K - 1.f) * Aminus * (newtonianVelocity.x - eigenvalues[0]));
#if DIM > 2
	eigenvectorsInverse[0 + NUM_STATES * 3] = -minMaxRowScale * (lorentzFactorSq * newtonianVelocity.z * (2.f * K - 1.f) * Aminus * (newtonianVelocity.x - eigenvalues[0]));
#endif
#endif
	eigenvectorsInverse[0 + NUM_STATES * (DIM+1)] = -minMaxRowScale * (-newtonianVelocity.x - lorentzFactorSq * (newtonianVelocitySq - newtonianVelocityXSq) * (2.f * K - 1.f) * (newtonianVelocity.x - Aminus * eigenvalues[0]) + K * Aminus * eigenvalues[0]);
	//mid normal row
	real midNormalRowScale = lorentzFactor * (K - 1.f);
	eigenvectorsInverse[1 + NUM_STATES * 0] = midNormalRowScale * (internalSpecificEnthalpy - lorentzFactor);
	eigenvectorsInverse[1 + NUM_STATES * 1] = midNormalRowScale * (lorentzFactor * newtonianVelocity.x);
#if DIM > 1
	eigenvectorsInverse[1 + NUM_STATES * 2] = midNormalRowScale * (lorentzFactor * newtonianVelocity.y);
#if DIM > 2
	eigenvectorsInverse[1 + NUM_STATES * 3] = midNormalRowScale * (lorentzFactor * newtonianVelocity.z);
#endif
#endif
	eigenvectorsInverse[1 + NUM_STATES * (DIM+1)] = midNormalRowScale * (-lorentzFactor);
	//mid tangent A row
#if DIM > 1
	real midTangentRowScale = 1.f / (internalSpecificEnthalpy * (1.f - newtonianVelocityXSq));
	eigenvectorsInverse[2 + NUM_STATES * 0] = midTangentRowScale * (-newtonianVelocity.y);
	eigenvectorsInverse[2 + NUM_STATES * 1] = midTangentRowScale * (newtonianVelocity.x * newtonianVelocity.y);
	eigenvectorsInverse[2 + NUM_STATES * 2] = midTangentRowScale * (1.f - newtonianVelocityXSq);
#if DIM > 2
	eigenvectorsInverse[2 + NUM_STATES * 3] = 0.f;
#endif
	eigenvectorsInverse[2 + NUM_STATES * (DIM+1)] = midTangentRowScale * (-newtonianVelocity.y);
#endif
	//mid tangent B row
#if DIM > 2
	eigenvectorsInverse[3 + NUM_STATES * 0] = midTangentRowScale * (-newtonianVelocity.z);
	eigenvectorsInverse[3 + NUM_STATES * 1] = midTangentRowScale * (newtonianVelocity.x * newtonianVelocity.z);
	eigenvectorsInverse[3 + NUM_STATES * 2] = 0.f;
	eigenvectorsInverse[3 + NUM_STATES * 3] = midTangentRowScale * (1.f - newtonianVelocityXSq);
	eigenvectorsInverse[3 + NUM_STATES * (DIM+1)] = midTangentRowScale * (-newtonianVelocity.z);
#endif
	//max row
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 0] =
		minMaxRowScale * (internalSpecificEnthalpy * lorentzFactor * Aplus * (newtonianVelocity.x - eigenvalues[DIM+1]) 
		- newtonianVelocity.x 
		- lorentzFactorSq * (newtonianVelocitySq - newtonianVelocityXSq) * (2.f * K - 1.f) * (newtonianVelocity.x - Aplus * eigenvalues[DIM+1]) 
		+ K * Aplus * eigenvalues[DIM+1]);
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 1] = minMaxRowScale * (1.f + lorentzFactorSq * (newtonianVelocitySq - newtonianVelocityXSq) * (2.f * K - 1.f) * (1.f - Aplus) - K * Aplus);
#if DIM > 1
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 2] = minMaxRowScale * (lorentzFactorSq * newtonianVelocity.y * (2.f * K - 1.f) * Aplus * (newtonianVelocity.x - eigenvalues[DIM+1]));
#if DIM > 2
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 3] = minMaxRowScale * (lorentzFactorSq * newtonianVelocity.z * (2.f * K - 1.f) * Aplus * (newtonianVelocity.x - eigenvalues[DIM+1]));
#endif
#endif
	eigenvectorsInverse[(DIM+1) + NUM_STATES * (DIM+1)] = minMaxRowScale * (-newtonianVelocity.x - lorentzFactorSq * (newtonianVelocitySq - newtonianVelocityXSq) * (2.f * K - 1.f) * (newtonianVelocity.x - Aplus * eigenvalues[DIM+1]) + K * Aplus * eigenvalues[DIM+1]);

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
			//a -90' rotation applied to the RHS of A must be corrected with a 90' rotation on the LHS of A
			//this rotates the elements of the column vectors by 90'
			//each column's x <- y, y <- -x
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = -eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = -eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z] = tmp;
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = -eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i] = tmp;
		}
	}
#endif
#endif

}

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* primitiveBuffer,
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
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, primitiveBuffer, stateBuffer, potentialBuffer, 0);
#if DIM > 1
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, primitiveBuffer, stateBuffer, potentialBuffer, 1);
#endif
#if DIM > 2
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, primitiveBuffer, stateBuffer, potentialBuffer, 2);
#endif
}



//HLLC based on
//"An HLLC Riemann Solver for Magnetohydrodynamics", Shengtai Li, 2003

#include "HydroGPU/Shared/Common.h"

void calcEigenvaluesSide(
	__global real* eigenvaluesBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	int side);

void calcEigenvaluesSide(
	__global real* eigenvaluesBuffer,
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
internalEnergyDensityL = max(0.f, internalEnergyDensityL);	//magnetic energy is exceeding total energy ...
	real pressureL = gammaMinusOne * internalEnergyDensityL;
	//real enthalpyTotalL = (totalHydroEnergyDensityL + pressureL) / densityL;
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
internalEnergyDensityR = max(0.f, internalEnergyDensityR);	//magnetic energy is exceeding total energy ...
	real pressureR = gammaMinusOne * internalEnergyDensityR;
	//real enthalpyTotalR = (totalHydroEnergyDensityR + pressureR) / densityR;
	real roeWeightR = sqrt(densityR);

	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real4 velocity = (velocityL * roeWeightL + velocityR * roeWeightR) * roeWeightNormalization;
	real4 magneticField = (magneticFieldL * roeWeightL + magneticFieldR * roeWeightR) * roeWeightNormalization;
	real pressure = (pressureL * roeWeightL + pressureR * roeWeightR) * roeWeightNormalization;
	//non-mhd hydro papers say to do this, but I get much greater dCons/dPrim eigenvector orthogonality error with this enthalpyTotal
	//real enthalpyTotal = (enthalpyTotalL * roeWeightL + enthalpyTotalR * roeWeightR) * roeWeightNormalization;
	real density = sqrt(densityL * densityR);
	
#if DIM > 1
	if (side == 1) {
		// -90' rotation to put the y axis contents into the x axis
		velocity.xy = velocity.yx;
		magneticField.xy = magneticField.yx;
	} 
#if DIM > 2
	else if (side == 2) {
		//-90' rotation to put the z axis in the x axis
		velocity.xz = velocity.zx;
		magneticField.xz = magneticField.zx;
	}
#endif
#endif

	real magneticFieldSq = dot(magneticField, magneticField);
	real sqrtDensity = sqrt(density);
	
	//matrices are stored as A_ij = A[i + height * j]

	real speedOfSoundSq = gamma * pressure / density;
	real AlfvenSpeed = fabs(magneticField.x) / (sqrtDensity * sqrtVaccuumPermeability);
	real AlfvenSpeedSq = AlfvenSpeed * AlfvenSpeed;
	
	real starSpeedSq = .5f * (speedOfSoundSq + magneticFieldSq / (density * vaccuumPermeability));
	real discr = starSpeedSq * starSpeedSq - speedOfSoundSq * AlfvenSpeedSq;
	real discrSqrt = sqrt(discr);
	real fastSpeedSq = starSpeedSq + discrSqrt;
	real fastSpeed = sqrt(fastSpeedSq);
	real slowSpeedSq = starSpeedSq - discrSqrt;
	real slowSpeed = sqrt(slowSpeedSq);
		
	eigenvalues[0] = velocity.x - fastSpeed;
	eigenvalues[1] = velocity.x - AlfvenSpeed;
	eigenvalues[2] = velocity.x - slowSpeed;
	eigenvalues[3] = velocity.x;
	eigenvalues[4] = velocity.x;
	eigenvalues[5] = velocity.x + slowSpeed;
	eigenvalues[6] = velocity.x + AlfvenSpeed;
	eigenvalues[7] = velocity.x + fastSpeed;
}

__kernel void calcEigenvalues(
	__global real* eigenvaluesBuffer,
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
	calcEigenvaluesSide(eigenvaluesBuffer, stateBuffer, potentialBuffer, 0);
#if DIM > 1
	calcEigenvaluesSide(eigenvaluesBuffer, stateBuffer, potentialBuffer, 1);
#endif
#if DIM > 2
	calcEigenvaluesSide(eigenvaluesBuffer, stateBuffer, potentialBuffer, 2);
#endif
}

//do we want to use interface wavespeeds to calculate cfl?
// esp when the flux values are computed from cell wavespeeds
__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real* eigenvaluesBuffer,
	real cfl)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) {
		cflBuffer[index] = INFINITY;
		return;
	}

	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int indexNext = index + stepsize[side];
		
		const __global real* eigenvaluesL = eigenvaluesBuffer + NUM_STATES * (side + DIM * index);
		const __global real* eigenvaluesR = eigenvaluesBuffer + NUM_STATES * (side + DIM * indexNext);

		real minLambda = 0.f;
		real maxLambda = 0.f;
		for (int i = 0; i < NUM_STATES; ++i) {	
			maxLambda = max(maxLambda, eigenvaluesL[i]);
			minLambda = min(minLambda, eigenvaluesR[i]);
		}

		real dum = dx[side] / (maxLambda - minLambda);
		result = min(result, dum);
	}
		
	cflBuffer[index] = cfl * result;
}

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	real dt_dx,
	int side);

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	real dt_dx,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;

	const __global real* srcStateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* srcStateR = stateBuffer + NUM_STATES * index;

	real stateL[NUM_STATES];
	real stateR[NUM_STATES];

	// rotate into x axis
	for (int i = 0; i < NUM_STATES; ++i) {
		stateL[i] = srcStateL[i];
		stateR[i] = srcStateR[i];
	}
	{
		real tmp;
		
		tmp = stateL[STATE_MOMENTUM_X];
		stateL[STATE_MOMENTUM_X] = stateL[STATE_MOMENTUM_X+side];
		stateL[STATE_MOMENTUM_X+side] = tmp;
		
		tmp = stateL[STATE_MAGNETIC_FIELD_X];
		stateL[STATE_MAGNETIC_FIELD_X] = stateL[STATE_MAGNETIC_FIELD_X+side];
		stateL[STATE_MAGNETIC_FIELD_X+side] = tmp;
		
		tmp = stateR[STATE_MOMENTUM_X];
		stateR[STATE_MOMENTUM_X] = stateR[STATE_MOMENTUM_X+side];
		stateR[STATE_MOMENTUM_X+side] = tmp;
		
		tmp = stateR[STATE_MAGNETIC_FIELD_X];
		stateR[STATE_MAGNETIC_FIELD_X] = stateR[STATE_MAGNETIC_FIELD_X+side];
		stateR[STATE_MAGNETIC_FIELD_X+side] = tmp;
	}

	__global real* flux = fluxBuffer + NUM_STATES * interfaceIndex;

	const real gammaMinusOne = gamma - 1.f;

	real densityL = stateL[STATE_DENSITY];
	real4 velocityL = VELOCITY(stateL);
	real4 magneticFieldL = (real4)(stateL[STATE_MAGNETIC_FIELD_X], stateL[STATE_MAGNETIC_FIELD_Y], stateL[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticFieldSqL = dot(magneticFieldL, magneticFieldL);
	real magneticEnergyDensityL = .5f * dot(magneticFieldL, magneticFieldL) / vaccuumPermeability;
	real totalPlasmaEnergyDensityL = stateL[STATE_ENERGY_TOTAL];
	real totalHydroEnergyDensityL = totalPlasmaEnergyDensityL - magneticEnergyDensityL;
	real kineticEnergyDensityL = .5f * densityL * dot(velocityL, velocityL);
	real potentialEnergyL = potentialBuffer[indexPrev];
	real potentialEnergyDensityL = densityL * potentialEnergyL; 
	real internalEnergyDensityL = totalHydroEnergyDensityL - kineticEnergyDensityL - potentialEnergyDensityL;
internalEnergyDensityL = max(0.f, internalEnergyDensityL);	//magnetic energy is exceeding total energy ...
	real pressureL = gammaMinusOne * internalEnergyDensityL;
	real enthalpyTotalL = (totalHydroEnergyDensityL + pressureL) / densityL;
	real speedOfSoundSqL = gamma * pressureL / densityL;
	real AlfvenSpeedSqL = magneticFieldL.x * magneticFieldL.x / (densityL * vaccuumPermeability);
	real starSpeedSqL = .5f * (speedOfSoundSqL + magneticFieldSqL / (densityL * vaccuumPermeability));
	real discrL = starSpeedSqL * starSpeedSqL - speedOfSoundSqL * AlfvenSpeedSqL;
	real discrSqrtL = sqrt(discrL);
	real fastSpeedSqL = starSpeedSqL + discrSqrtL;
	real fastSpeedL = sqrt(fastSpeedSqL);
	real roeWeightL = sqrt(densityL);

	real densityR = stateR[STATE_DENSITY];
	real4 velocityR = VELOCITY(stateR);
	real4 magneticFieldR = (real4)(stateR[STATE_MAGNETIC_FIELD_X], stateR[STATE_MAGNETIC_FIELD_Y], stateR[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticFieldSqR = dot(magneticFieldR, magneticFieldR);
	real magneticEnergyDensityR = .5f * dot(magneticFieldR, magneticFieldR) / vaccuumPermeability;
	real totalPlasmaEnergyDensityR = stateR[STATE_ENERGY_TOTAL];
	real totalHydroEnergyDensityR = totalPlasmaEnergyDensityR - magneticEnergyDensityR;
	real kineticEnergyDensityR = .5f * densityR * dot(velocityR, velocityR);
	real potentialEnergyR = potentialBuffer[index];
	real potentialEnergyDensityR = densityR * potentialEnergyR;
	real internalEnergyDensityR = totalHydroEnergyDensityR - kineticEnergyDensityR - potentialEnergyDensityR;
internalEnergyDensityR = max(0.f, internalEnergyDensityR);	//magnetic energy is exceeding total energy ...
	real pressureR = gammaMinusOne * internalEnergyDensityR;
	real enthalpyTotalR = (totalHydroEnergyDensityR + pressureR) / densityR;
	real speedOfSoundSqR = gamma * pressureR / densityR;
	real AlfvenSpeedSqR = magneticFieldR.x * magneticFieldR.x / (densityR * vaccuumPermeability);
	real starSpeedSqR = .5f * (speedOfSoundSqR + magneticFieldSqR / (densityR * vaccuumPermeability));
	real discrR = starSpeedSqR * starSpeedSqR - speedOfSoundSqR * AlfvenSpeedSqR;
	real discrSqrtR = sqrt(discrR);
	real fastSpeedSqR = starSpeedSqR + discrSqrtR;
	real fastSpeedR = sqrt(fastSpeedSqR);
	real roeWeightR = sqrt(densityR);

	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real4 velocity = (velocityL * roeWeightL + velocityR * roeWeightR) * roeWeightNormalization;
	real4 magneticField = (magneticFieldL * roeWeightL + magneticFieldR * roeWeightR) * roeWeightNormalization;
	real pressure = (pressureL * roeWeightL + pressureR * roeWeightR) * roeWeightNormalization;
	//non-mhd hydro papers say to do this, but I get much greater dCons/dPrim eigenvector orthogonality error with this enthalpyTotal
	//real enthalpyTotal = (enthalpyTotalL * roeWeightL + enthalpyTotalR * roeWeightR) * roeWeightNormalization;
	real density = sqrt(densityL * densityR);
	real magneticFieldSq = dot(magneticField, magneticField);
	
	real sqrtDensity = sqrt(density);
	
	//matrices are stored as A_ij = A[i + height * j]

	real speedOfSoundSq = gamma * pressure / density;
	real AlfvenSpeed = fabs(magneticField.x) / (sqrtDensity * sqrtVaccuumPermeability);
	real AlfvenSpeedSq = AlfvenSpeed * AlfvenSpeed;
	real starSpeedSq = .5f * (speedOfSoundSq + magneticFieldSq / (density * vaccuumPermeability));
	real discr = starSpeedSq * starSpeedSq - speedOfSoundSq * AlfvenSpeedSq;
	real discrSqrt = sqrt(discr);
	real fastSpeedSq = starSpeedSq + discrSqrt;
	real fastSpeed = sqrt(fastSpeedSq);

	real eigenvaluesMinL = velocityL.x - fastSpeedL;
	real eigenvaluesMaxR = velocityR.x + fastSpeedR;

	real eigenvaluesMin = velocity.x - fastSpeed;
	real eigenvaluesMax = velocity.x + fastSpeed;

	//flux

#if 0	//Davis direct
	real sl = eigenvaluesMinL;
	real sr = eigenvaluesMaxR;
#endif
#if 1	//Davis direct bounded
	real sl = min(eigenvaluesMinL, eigenvaluesMin);
	real sr = max(eigenvaluesMaxR, eigenvaluesMax);
#endif

	real vDotBL = dot(velocityL, magneticFieldL);
	real fluxL[NUM_STATES];
	fluxL[STATE_DENSITY] = densityL * velocityL.x;
	fluxL[STATE_MOMENTUM_X] = densityL * velocityL.x * velocityL.x - magneticFieldL.x * magneticFieldL.x / vaccuumPermeability +  pressureL + .5f * magneticFieldSqL;
	fluxL[STATE_MOMENTUM_Y] = densityL * velocityL.x * velocityL.y - magneticFieldL.x * magneticFieldL.y / vaccuumPermeability;
	fluxL[STATE_MOMENTUM_Z] = densityL * velocityL.x * velocityL.z - magneticFieldL.x * magneticFieldL.z / vaccuumPermeability;
	fluxL[STATE_MAGNETIC_FIELD_X] = 0.f;
	fluxL[STATE_MAGNETIC_FIELD_Y] = velocityL.x * magneticFieldL.y - magneticFieldL.x * velocityL.y;
	fluxL[STATE_MAGNETIC_FIELD_Z] = velocityL.x * magneticFieldL.z - magneticFieldL.x * velocityL.z;
	fluxL[STATE_ENERGY_TOTAL] = densityL * (enthalpyTotalL + .5f * magneticFieldSqL) * velocityL.x - vDotBL * magneticFieldL.x / vaccuumPermeability;
	
	real vDotBR = dot(velocityR, magneticFieldR);
	real fluxR[NUM_STATES];
	fluxR[STATE_DENSITY] = densityR * velocityR.x;
	fluxR[STATE_MOMENTUM_X] = densityR * velocityR.x * velocityR.x - magneticFieldR.x * magneticFieldR.x / vaccuumPermeability +  pressureR + .5f * magneticFieldSqR;
	fluxR[STATE_MOMENTUM_Y] = densityR * velocityR.x * velocityR.y - magneticFieldR.x * magneticFieldR.y / vaccuumPermeability;
	fluxR[STATE_MOMENTUM_Z] = densityR * velocityR.x * velocityR.z - magneticFieldR.x * magneticFieldR.z / vaccuumPermeability;
	fluxR[STATE_MAGNETIC_FIELD_X] = 0.f;
	fluxR[STATE_MAGNETIC_FIELD_Y] = velocityR.x * magneticFieldR.y - magneticFieldR.x * velocityR.y;
	fluxR[STATE_MAGNETIC_FIELD_Z] = velocityR.x * magneticFieldR.z - magneticFieldR.x * velocityR.z;
	fluxR[STATE_ENERGY_TOTAL] = densityR * (enthalpyTotalR + .5f * magneticFieldSqR) * velocityR.x - vDotBR * magneticFieldR.x / vaccuumPermeability;
	

	real sStar = (densityR * velocityR.x * (sr - velocityR.x) - densityL * velocityL.x * (sl - velocityL.x) + pressureL - pressureR - magneticFieldL.x * magneticFieldL.x + magneticFieldR.x * magneticFieldR.x) 
					/ (densityR * (sr - velocityR.x) - densityL * (sl - velocityL.x));
	
	if (0 <= sl) {
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxL[i];
		}
	} else if (sl <= 0.f && 0.f <= sStar) {
		
		real4 magneticFieldStar;
		magneticFieldStar.x = (sr * magneticFieldR.x - sl * magneticFieldL.x) / (sr - sl);
		magneticFieldStar.y = ((sl - velocityL.x - magneticFieldL.x * magneticFieldStar.x / (densityL * (sl - velocityL.x))) * magneticFieldL.y - (magneticFieldStar.x - magneticFieldL.x) * velocityL.y) / (sl - sStar - magneticFieldStar.x * magneticFieldStar.x / (densityL * (sl - velocityL.x)));
		magneticFieldStar.z = ((sl - velocityL.x - magneticFieldL.x * magneticFieldStar.x / (densityL * (sl - velocityL.x))) * magneticFieldL.z - (magneticFieldStar.x - magneticFieldL.x) * velocityL.z) / (sl - sStar - magneticFieldStar.x * magneticFieldStar.x / (densityL * (sl - velocityL.x)));
		magneticFieldStar.w = 0.f;
		real pressureStar = densityL * (sl - velocityL.x) * (sStar - velocityL.x) + pressureL + .5f * magneticFieldSqL - magneticFieldL.x * magneticFieldL.x + magneticFieldStar.x * magneticFieldStar.x;
		
		real stateLStar[NUM_STATES];
		stateLStar[STATE_DENSITY] = densityL * (sl - velocityL.x) / (sl - sStar);
		stateLStar[STATE_MOMENTUM_X] = stateLStar[STATE_DENSITY] * sStar;
		stateLStar[STATE_MOMENTUM_Y] = stateLStar[STATE_DENSITY] * velocityL.y - (magneticFieldStar.x * magneticFieldStar.y - magneticFieldL.x * magneticFieldL.y) / (sl - sStar);
		stateLStar[STATE_MOMENTUM_Z] = stateLStar[STATE_DENSITY] * velocityL.z - (magneticFieldStar.x * magneticFieldStar.z - magneticFieldL.x * magneticFieldL.z) / (sl - sStar);
		real4 velocityStar = (real4)(stateLStar[STATE_MOMENTUM_X], stateLStar[STATE_MOMENTUM_Y], stateLStar[STATE_MOMENTUM_Z], 0.f) / stateLStar[STATE_DENSITY];
		real vDotBStar = dot(magneticFieldStar, velocityStar);
		stateLStar[STATE_MAGNETIC_FIELD_X] = magneticFieldStar.x;
		stateLStar[STATE_MAGNETIC_FIELD_Y] = magneticFieldStar.y;
		stateLStar[STATE_MAGNETIC_FIELD_Z] = magneticFieldStar.z;
		stateLStar[STATE_ENERGY_TOTAL] = (stateL[STATE_ENERGY_TOTAL] * (sl - velocityL.x) + (pressureStar * sStar - pressureL * velocityL.x) - (magneticFieldStar.x * vDotBStar - magneticFieldL.x * vDotBL)) / (sl - sStar);
		
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxL[i] + sl * (stateLStar[i] - stateL[i]);
		}
	
	} else if (sStar <= 0.f && 0.f <= sr) {
		
		real4 magneticFieldStar;
		magneticFieldStar.x = (sr * magneticFieldR.x - sl * magneticFieldL.x) / (sr - sl);
		magneticFieldStar.y = ((sl - velocityR.x - magneticFieldR.x * magneticFieldStar.x / (densityR * (sl - velocityR.x))) * magneticFieldR.y - (magneticFieldStar.x - magneticFieldR.x) * velocityR.y) / (sl - sStar - magneticFieldStar.x * magneticFieldStar.x / (densityR * (sl - velocityR.x)));
		magneticFieldStar.z = ((sl - velocityR.x - magneticFieldR.x * magneticFieldStar.x / (densityR * (sl - velocityR.x))) * magneticFieldR.z - (magneticFieldStar.x - magneticFieldR.x) * velocityR.z) / (sl - sStar - magneticFieldStar.x * magneticFieldStar.x / (densityR * (sl - velocityR.x)));
		magneticFieldStar.w = 0.f;
		real pressureStar = densityR * (sl - velocityR.x) * (sStar - velocityR.x) + pressureR + .5f * magneticFieldSqR - magneticFieldR.x * magneticFieldR.x + magneticFieldStar.x * magneticFieldStar.x;
		
		real stateRStar[NUM_STATES];
		stateRStar[STATE_DENSITY] = densityR * (sr - velocityR.x) / (sr - sStar);
		stateRStar[STATE_MOMENTUM_X] = stateRStar[STATE_DENSITY] * sStar;
		stateRStar[STATE_MOMENTUM_Y] = stateRStar[STATE_DENSITY] * velocityR.y - (magneticFieldStar.x * magneticFieldStar.y - magneticFieldR.x * magneticFieldR.y) / (sr - sStar);
		stateRStar[STATE_MOMENTUM_Z] = stateRStar[STATE_DENSITY] * velocityR.z - (magneticFieldStar.x * magneticFieldStar.z - magneticFieldR.x * magneticFieldR.z) / (sr - sStar);
		real4 velocityStar = (real4)(stateRStar[STATE_MOMENTUM_X], stateRStar[STATE_MOMENTUM_Y], stateRStar[STATE_MOMENTUM_Z], 0.f) / stateRStar[STATE_DENSITY];
		real vDotBStar = dot(magneticFieldStar, velocityStar);
		stateRStar[STATE_MAGNETIC_FIELD_X] = magneticFieldStar.x;
		stateRStar[STATE_MAGNETIC_FIELD_Y] = magneticFieldStar.y;
		stateRStar[STATE_MAGNETIC_FIELD_Z] = magneticFieldStar.z;
		stateRStar[STATE_ENERGY_TOTAL] = (stateR[STATE_ENERGY_TOTAL] * (sr - velocityR.x) + (pressureStar * sStar - pressureR * velocityR.x) - (magneticFieldStar.x * vDotBStar - magneticFieldR.x * vDotBR)) / (sr - sStar);
		
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxR[i] + sr * (stateRStar[i] - stateR[i]);
		}
	
	} else if (sr <= 0.f) {
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxR[i];
		}
	}


/*
slope limiter?

theta = sign of eigenvalues

fhalf = 1/2 (fR + fL) - 1/2 (theta + phi (epsilon - theta)) (fR - fL)

donor cell when phi = 0:

fhalf = 1/2 (fR + fL) - 1/2 theta (fR - fL)
... for theta = 1: 	fhalf = 1/2 fR + 1/2 fL - 1/2 fR + 1/2 fL = fL
... for theta = -1:	fhalf = 1/2 fR + 1/2 fL + 1/2 fR - 1/2 fL = fR

... for phi = slope of rTile
			= for theta = 1:	delta q(i-1) / delta q(i)
			= for theta = -1:	delta q(i+1) / delta q(i)

so what do we use to consider delta q?  the states themselves, or any sort of transformation (as Roe does)?
how about (by Hydrodynamics ii) delta q = delta f / lambda
so what is delta f?  
Hydrodynamics ii stops at mentioning the deconstruction of delta q into delta q- and delta q+
 and its association with lambda+ and lambda-
 which are the eigenvalue min and max used in the flux calculation.
 is this the slope that should be used with the slope limiter?
or can we use delta q- and delta q+ for the lhs and rhs of the delta q slope, choosing one or the other based on the velocity sign 
*/
#if 0
	for (int i = 0; i < NUM_STATES; ++i) {
		real eigenvalue = eigenvalues[i];
		real deltaFlux = fluxR[i] - fluxL[i];
		real deltaQ = stateR[i] - stateL[i];
		real rTilde = deltaFlux / deltaQ;
		real theta;
		if (eigenvalue >= 0.f) {
			theta = 1.f;
			//rTilde = (stateMid - stateL[i]) / deltaQ;
		} else {
			theta = -1.f;
			//rTilde = (stateR[i] - stateMid) / deltaQ;
		}
		real phi = slopeLimiter(rTilde);
		real epsilon = eigenvalue * dt_dx;
		flux[i] -= .5f * deltaFlux * (theta + phi * (epsilon - theta) / (float)DIM);
	}
#endif

	//rotate back to side
	{
		real tmp;

		tmp = flux[STATE_MOMENTUM_X];
		flux[STATE_MOMENTUM_X] = flux[STATE_MOMENTUM_X + side];
		flux[STATE_MOMENTUM_X + side] = tmp;
		
		tmp = flux[STATE_MAGNETIC_FIELD_X];
		flux[STATE_MAGNETIC_FIELD_X] = flux[STATE_MAGNETIC_FIELD_X + side];
		flux[STATE_MAGNETIC_FIELD_X + side] = tmp;
	}
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global real* dtBuffer)
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
	
	real dt = dtBuffer[0];
	
	calcFluxSide(fluxBuffer, stateBuffer, potentialBuffer, dt/DX, 0);
#if DIM > 1
	calcFluxSide(fluxBuffer, stateBuffer, potentialBuffer, dt/DY, 1);
#endif
#if DIM > 2
	calcFluxSide(fluxBuffer, stateBuffer, potentialBuffer, dt/DZ, 2);
#endif
}

__kernel void calcFluxDeriv(
	__global real* derivBuffer,
	const __global real* fluxBuffer)
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

	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexNext = index + stepsize[side];
		const __global real* fluxL = fluxBuffer + NUM_STATES * (side + DIM * index);
		const __global real* fluxR = fluxBuffer + NUM_STATES * (side + DIM * indexNext);
		for (int j = 0; j < NUM_STATES; ++j) {
			real deltaFlux = fluxR[j] - fluxL[j];
			deriv[j] -= deltaFlux / dx[side];
		}
	}
}



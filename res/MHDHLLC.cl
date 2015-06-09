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

	real8 stateL = rotateStateToX(stateBuffer + NUM_STATES * indexPrev, side);
	real8 stateR = rotateStateToX(stateBuffer + NUM_STATES * index, side);

	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;

	Primitives_t primsL = calcPrimitivesFromState(stateL, potentialBuffer[indexPrev]);
	const real roeWeightL = 1.f;//sqrt(densityL);

	Primitives_t primsR = calcPrimitivesFromState(stateR, potentialBuffer[index]);
	const real roeWeightR = 1.f;//sqrt(primsR.density);

	const real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	Primitives_t primsAvg;
	//non-mhd hydro papers say to do this, but I get much greater dCons/dPrim eigenvector orthogonality error with this enthalpyTotal
	//real enthalpyTotal = (enthalpyTotalL * roeWeightL + primsR.enthalpyTotal * roeWeightR) * roeWeightNormalization;
	//real density = sqrt(densityL * primsR.density);
	primsAvg.density = (primsL.density * roeWeightL + primsR.density * roeWeightR) * roeWeightNormalization;
	primsAvg.pressure = (primsL.pressure * roeWeightL + primsR.pressure * roeWeightR) * roeWeightNormalization;
	primsAvg.pressureTotal = (primsL.pressureTotal * roeWeightL + primsR.pressureTotal * roeWeightR) * roeWeightNormalization;
	primsAvg.velocity = (primsL.velocity * roeWeightL + primsR.velocity * roeWeightR) * roeWeightNormalization;
	primsAvg.magneticField = (primsL.magneticField * roeWeightL + primsR.magneticField * roeWeightR) * roeWeightNormalization;

	Wavespeed_t speedL = calcWavespeedFromPrimitives(primsL);
	Wavespeed_t speedR = calcWavespeedFromPrimitives(primsR);
	Wavespeed_t speedAvg = calcWavespeedFromPrimitives(primsAvg);

	//the Euler HLLC calculates the left and right wavespeeds here
	// as the min of the left and avg eigenvalue; and the max of the middle and right wavespeeds
	// ... which are the only wavespeeds considered by the timestep solver

	//here I'm testing on an individual basis ...
	// I'm pretty sure only the min and max are used anyways
	eigenvalues[0] = min(primsL.velocity.x - speedL.fast, primsAvg.velocity.x - speedAvg.fast);
	eigenvalues[1] = min(primsL.velocity.x - speedL.Alfven, primsAvg.velocity.x - speedAvg.Alfven);
	eigenvalues[2] = min(primsL.velocity.x - speedL.slow, primsAvg.velocity.x - speedAvg.slow);
	eigenvalues[3] = primsAvg.velocity.x;
	eigenvalues[4] = primsAvg.velocity.x;
	eigenvalues[5] = max(primsR.velocity.x + speedR.slow, primsAvg.velocity.x + speedAvg.slow);
	eigenvalues[6] = max(primsR.velocity.x + speedR.Alfven, primsAvg.velocity.x + speedAvg.Alfven);
	eigenvalues[7] = max(primsR.velocity.x + speedR.fast, primsAvg.velocity.x + speedAvg.fast);
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

		real minLambda = min(0.f, eigenvaluesR[0]);
		real maxLambda = max(0.f, eigenvaluesL[NUM_STATES-1]);

		real dum = dx[side] / (maxLambda - minLambda);
		result = min(result, dum);
	}
		
	cflBuffer[index] = cfl * result;
}

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* potentialBuffer,
	real dt_dx,
	int side);

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* potentialBuffer,
	real dt_dx,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;

	real8 stateL = rotateStateToX(stateBuffer + NUM_STATES * indexPrev, side);
	real8 stateR = rotateStateToX(stateBuffer + NUM_STATES * index, side);

	const __global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	
	__global real* flux = fluxBuffer + NUM_STATES * interfaceIndex;

	Primitives_t primsL = calcPrimitivesFromState(stateL, potentialBuffer[indexPrev]);
	Wavespeed_t speedL = calcWavespeedFromPrimitives(primsL);
	real magneticFieldSqL = dot(primsL.magneticField, primsL.magneticField);
	const real roeWeightL = 1.f;//sqrt(primsL.density);

	Primitives_t primsR = calcPrimitivesFromState(stateR, potentialBuffer[index]);
	Wavespeed_t speedR = calcWavespeedFromPrimitives(primsR);
	real magneticFieldSqR = dot(primsR.magneticField, primsR.magneticField);
	const real roeWeightR = 1.f;//sqrt(primsR.density);

	const real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	Primitives_t primsAvg;
	primsAvg.density = (primsL.density * roeWeightL + primsR.density * roeWeightR) * roeWeightNormalization;
	primsAvg.pressure = (primsL.pressure * roeWeightL + primsR.pressure * roeWeightR) * roeWeightNormalization;
	primsAvg.pressureTotal = (primsL.pressureTotal * roeWeightL + primsR.pressureTotal * roeWeightR) * roeWeightNormalization;
	primsAvg.velocity = (primsL.velocity * roeWeightL + primsR.velocity * roeWeightR) * roeWeightNormalization;
	primsAvg.magneticField = (primsL.magneticField * roeWeightL + primsR.magneticField * roeWeightR) * roeWeightNormalization;
	Wavespeed_t speedAvg = calcWavespeedFromPrimitives(primsAvg);

	real eigenvaluesMinL = primsL.velocity.x - speedL.fast;
	real eigenvaluesMaxR = primsR.velocity.x + speedR.fast;
	real eigenvaluesMin = primsAvg.velocity.x - speedAvg.fast;
	real eigenvaluesMax = primsAvg.velocity.x + speedAvg.fast;

	//flux

	//real sl = eigenvalues[0];
	//real sr = eigenvalues[DIM+1];
	real sl = min(eigenvaluesMinL, eigenvaluesMin);
	real sr = max(eigenvaluesMaxR, eigenvaluesMax);

	real vDotBL = dot(primsL.velocity, primsL.magneticField);
	real fluxL[NUM_STATES];
	fluxL[STATE_DENSITY] = primsL.density * primsL.velocity.x;
	fluxL[STATE_MOMENTUM_X] = primsL.density * primsL.velocity.x * primsL.velocity.x - primsL.magneticField.x * primsL.magneticField.x / vaccuumPermeability +  primsL.pressureTotal + .5f * magneticFieldSqL;
	fluxL[STATE_MOMENTUM_Y] = primsL.density * primsL.velocity.x * primsL.velocity.y - primsL.magneticField.x * primsL.magneticField.y / vaccuumPermeability;
	fluxL[STATE_MOMENTUM_Z] = primsL.density * primsL.velocity.x * primsL.velocity.z - primsL.magneticField.x * primsL.magneticField.z / vaccuumPermeability;
	fluxL[STATE_MAGNETIC_FIELD_X] = 0.f;
	fluxL[STATE_MAGNETIC_FIELD_Y] = primsL.velocity.x * primsL.magneticField.y - primsL.magneticField.x * primsL.velocity.y;
	fluxL[STATE_MAGNETIC_FIELD_Z] = primsL.velocity.x * primsL.magneticField.z - primsL.magneticField.x * primsL.velocity.z;
	fluxL[STATE_ENERGY_TOTAL] = primsL.density * (primsL.enthalpyTotal + .5f * magneticFieldSqL) * primsL.velocity.x - vDotBL * primsL.magneticField.x / vaccuumPermeability;
	
	real vDotBR = dot(primsR.velocity, primsR.magneticField);
	real fluxR[NUM_STATES];
	fluxR[STATE_DENSITY] = primsR.density * primsR.velocity.x;
	fluxR[STATE_MOMENTUM_X] = primsR.density * primsR.velocity.x * primsR.velocity.x - primsR.magneticField.x * primsR.magneticField.x / vaccuumPermeability +  primsR.pressureTotal + .5f * magneticFieldSqR;
	fluxR[STATE_MOMENTUM_Y] = primsR.density * primsR.velocity.x * primsR.velocity.y - primsR.magneticField.x * primsR.magneticField.y / vaccuumPermeability;
	fluxR[STATE_MOMENTUM_Z] = primsR.density * primsR.velocity.x * primsR.velocity.z - primsR.magneticField.x * primsR.magneticField.z / vaccuumPermeability;
	fluxR[STATE_MAGNETIC_FIELD_X] = 0.f;
	fluxR[STATE_MAGNETIC_FIELD_Y] = primsR.velocity.x * primsR.magneticField.y - primsR.magneticField.x * primsR.velocity.y;
	fluxR[STATE_MAGNETIC_FIELD_Z] = primsR.velocity.x * primsR.magneticField.z - primsR.magneticField.x * primsR.velocity.z;
	fluxR[STATE_ENERGY_TOTAL] = primsR.density * (primsR.enthalpyTotal + .5f * magneticFieldSqR) * primsR.velocity.x - vDotBR * primsR.magneticField.x / vaccuumPermeability;
	

	real sStar = (primsR.density * primsR.velocity.x * (sr - primsR.velocity.x) - primsL.density * primsL.velocity.x * (sl - primsL.velocity.x) + primsL.pressureTotal - primsR.pressureTotal - primsL.magneticField.x * primsL.magneticField.x + primsR.magneticField.x * primsR.magneticField.x) 
					/ (primsR.density * (sr - primsR.velocity.x) - primsL.density * (sl - primsL.velocity.x));
	
	if (0.f <= sl) {
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxL[i];
		}
#if 0	//HLL
	} else if (sl <= 0.f && 0.f <= sr) {
		//(sr * fluxL[j] - sl * fluxR[j] + sl * sr * (stateR[j] - stateL[j])) / (sr - sl)
		real invDenom = 1.f / (sr - sl);
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = (sr * fluxL[i] - sl * fluxR[i] + sl * sr * (stateR[i] - stateL[i])) * invDenom; 
		}
		
#else	//HLLC
	} else if (sl <= 0.f && 0.f <= sStar) {
		
		real4 magneticFieldStar;
		magneticFieldStar.x = (sr * primsR.magneticField.x - sl * primsL.magneticField.x) / (sr - sl);
		magneticFieldStar.y = ((sl - primsL.velocity.x - primsL.magneticField.x * magneticFieldStar.x / (primsL.density * (sl - primsL.velocity.x))) * primsL.magneticField.y 
			- (magneticFieldStar.x - primsL.magneticField.x) * primsL.velocity.y) / (sl - sStar - magneticFieldStar.x * magneticFieldStar.x / (primsL.density * (sl - primsL.velocity.x)));
		magneticFieldStar.z = ((sl - primsL.velocity.x - primsL.magneticField.x * magneticFieldStar.x / (primsL.density * (sl - primsL.velocity.x))) * primsL.magneticField.z 
			- (magneticFieldStar.x - primsL.magneticField.x) * primsL.velocity.z) / (sl - sStar - magneticFieldStar.x * magneticFieldStar.x / (primsL.density * (sl - primsL.velocity.x)));
		magneticFieldStar.w = 0.f;
		real pressureStar = primsL.density * (sl - primsL.velocity.x) * (sStar - primsL.velocity.x) + primsL.pressureTotal + .5f * magneticFieldSqL - primsL.magneticField.x * primsL.magneticField.x + magneticFieldStar.x * magneticFieldStar.x;
		
		real stateLStar[NUM_STATES];
		stateLStar[STATE_DENSITY] = primsL.density * (sl - primsL.velocity.x) / (sl - sStar);
		stateLStar[STATE_MOMENTUM_X] = stateLStar[STATE_DENSITY] * sStar;
		stateLStar[STATE_MOMENTUM_Y] = stateLStar[STATE_DENSITY] * primsL.velocity.y - (magneticFieldStar.x * magneticFieldStar.y - primsL.magneticField.x * primsL.magneticField.y) / (sl - sStar);
		stateLStar[STATE_MOMENTUM_Z] = stateLStar[STATE_DENSITY] * primsL.velocity.z - (magneticFieldStar.x * magneticFieldStar.z - primsL.magneticField.x * primsL.magneticField.z) / (sl - sStar);
		real4 velocityStar = (real4)(stateLStar[STATE_MOMENTUM_X], stateLStar[STATE_MOMENTUM_Y], stateLStar[STATE_MOMENTUM_Z], 0.f) / stateLStar[STATE_DENSITY];
		real vDotBStar = dot(magneticFieldStar, velocityStar);
		stateLStar[STATE_MAGNETIC_FIELD_X] = magneticFieldStar.x;
		stateLStar[STATE_MAGNETIC_FIELD_Y] = magneticFieldStar.y;
		stateLStar[STATE_MAGNETIC_FIELD_Z] = magneticFieldStar.z;
		stateLStar[STATE_ENERGY_TOTAL] = stateLStar[STATE_DENSITY] * stateL[STATE_ENERGY_TOTAL] / primsL.density + ((pressureStar * sStar - primsL.pressureTotal * primsL.velocity.x) - (magneticFieldStar.x * vDotBStar - primsL.magneticField.x * vDotBL)) / (sl - sStar);
		//hydro-only
		//stateLStar[STATE_ENERGY_TOTAL] = stateLStar[STATE_DENSITY] * (energyTotalL + (sStar - primsL.velocity.x) * (sStar + primsL.pressureTotal / (primsL.density * (sl - primsL.velocity.x))));
		
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxL[i] + sl * (stateLStar[i] - stateL[i]);
		}
	
	} else if (sStar <= 0.f && 0.f <= sr) {
		
		real4 magneticFieldStar;
		magneticFieldStar.x = (sr * primsR.magneticField.x - sl * primsL.magneticField.x) / (sr - sl);
		magneticFieldStar.y = ((sr - primsR.velocity.x - primsR.magneticField.x * magneticFieldStar.x / (primsR.density * (sr - primsR.velocity.x))) * primsR.magneticField.y 
			- (magneticFieldStar.x - primsR.magneticField.x) * primsR.velocity.y) / (sr - sStar - magneticFieldStar.x * magneticFieldStar.x / (primsR.density * (sr - primsR.velocity.x)));
		magneticFieldStar.z = ((sr - primsR.velocity.x - primsR.magneticField.x * magneticFieldStar.x / (primsR.density * (sr - primsR.velocity.x))) * primsR.magneticField.z 
			- (magneticFieldStar.x - primsR.magneticField.x) * primsR.velocity.z) / (sr - sStar - magneticFieldStar.x * magneticFieldStar.x / (primsR.density * (sr - primsR.velocity.x)));
		magneticFieldStar.w = 0.f;
		real pressureStar = primsR.density * (sr - primsR.velocity.x) * (sStar - primsR.velocity.x) + primsR.pressureTotal + .5f * magneticFieldSqR - primsR.magneticField.x * primsR.magneticField.x + magneticFieldStar.x * magneticFieldStar.x;
		
		real stateRStar[NUM_STATES];
		stateRStar[STATE_DENSITY] = primsR.density * (sr - primsR.velocity.x) / (sr - sStar);
		stateRStar[STATE_MOMENTUM_X] = stateRStar[STATE_DENSITY] * sStar;
		stateRStar[STATE_MOMENTUM_Y] = stateRStar[STATE_DENSITY] * primsR.velocity.y - (magneticFieldStar.x * magneticFieldStar.y - primsR.magneticField.x * primsR.magneticField.y) / (sr - sStar);
		stateRStar[STATE_MOMENTUM_Z] = stateRStar[STATE_DENSITY] * primsR.velocity.z - (magneticFieldStar.x * magneticFieldStar.z - primsR.magneticField.x * primsR.magneticField.z) / (sr - sStar);
		real4 velocityStar = (real4)(stateRStar[STATE_MOMENTUM_X], stateRStar[STATE_MOMENTUM_Y], stateRStar[STATE_MOMENTUM_Z], 0.f) / stateRStar[STATE_DENSITY];
		real vDotBStar = dot(magneticFieldStar, velocityStar);
		stateRStar[STATE_MAGNETIC_FIELD_X] = magneticFieldStar.x;
		stateRStar[STATE_MAGNETIC_FIELD_Y] = magneticFieldStar.y;
		stateRStar[STATE_MAGNETIC_FIELD_Z] = magneticFieldStar.z;
		stateRStar[STATE_ENERGY_TOTAL] = stateRStar[STATE_DENSITY] * stateR[STATE_ENERGY_TOTAL] / primsR.density + ((pressureStar * sStar - primsR.pressureTotal * primsR.velocity.x) - (magneticFieldStar.x * vDotBStar - primsR.magneticField.x * vDotBR)) / (sr - sStar);
		
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxR[i] + sr * (stateRStar[i] - stateR[i]);
		}
#endif	
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
	const __global real* eigenvaluesBuffer,
	const __global real* potentialBuffer,
	real dt)
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
	
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, potentialBuffer, dt/DX, 0);
#if DIM > 1
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, potentialBuffer, dt/DY, 1);
#endif
#if DIM > 2
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, potentialBuffer, dt/DZ, 2);
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



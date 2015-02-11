#include "HydroGPU/Shared/Common.h"

/*
MHD Burgers:

    [  rho  ]             [  rho  ]              [    0   ]    [         0        ]
  d [rho v_i]     d       [rho v_i]      d       [  -B_i  ]    [     dP* / dx_i   ]
 -- [  B_i  ] + ---- (v_j [  B_i  ]) + ---- (B_j [  -v_i  ]) + [         0        ] = 0
 dt [   E   ]   dx_j      [   E   ]    dx_j      [-v_k B_k]    [ d(P* v_j) / dx_j ]

for E = rho e + 1/2 B^2
e = total specific energy = eKin + eInt = 1/2 v^2 + eInt
and P* = P + 1/2 B^2
P = (gamma - 1) rho eInt

Looks a lot like the Euler Burgers breakdown, except with the added B advection term
*/

//based on max inter-cell wavespeed
__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	real cfl)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		cflBuffer[index] = INFINITY;
		return;
	}

	const __global real* state = stateBuffer + NUM_STATES * index;

	real density = state[STATE_DENSITY];
	real energyTotal = state[STATE_ENERGY_TOTAL];
	real4 velocity = (real4)(state[STATE_MOMENTUM_X], state[STATE_MOMENTUM_Y], state[STATE_MOMENTUM_Z], 0.f) / density;
	real4 magneticField = (real4)(state[STATE_MAGNETIC_FIELD_X], state[STATE_MAGNETIC_FIELD_Y], state[STATE_MAGNETIC_FIELD_Z], 0.f);
	
	real velocitySq = dot(velocity, velocity);
	real magneticFieldSq = dot(magneticField, magneticField);
	
	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5f * velocitySq;
	real specificEnergyMagnetic = .5f * magneticFieldSq;
	real specificEnergyPotential = potentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential - specificEnergyMagnetic;

	real speedOfSoundSq = gamma * (gamma - 1.f) * specificEnergyInternal;

	//use fast speed for timestep
	real tmp1 = speedOfSoundSq + magneticFieldSq / density;
	
	//subtract magnetic field along normal
	//real discr = tmp1 * tmp1 - 4.f * speedOfSoundSq * magneticFieldXSq / density;
	//real tmp2 = sqrt(discr);
	//real fastSpeedSq = .5f * (tmp1 + tmp2);
	//...or assume it to be zeor
	real fastSpeedSq = tmp1;	
	
	real fastSpeed = sqrt(fastSpeedSq);

	real result = dx[0] / (fastSpeed + fabs(velocity[0]));
	for (int side = 1; side < DIM; ++side) {
		real dum = dx[side] / (fastSpeed + fabs(velocity[side]));
		result = min(result, dum);
	}
	
	cflBuffer[index] = cfl * result;
}

__kernel void calcInterfaceVelocity(
	__global real* interfaceVelocityBuffer,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1 
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	int indexR = index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		
		real densityL = stateBuffer[STATE_DENSITY + NUM_STATES * indexL];
		real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL] / densityL;
		
		real densityR = stateBuffer[STATE_DENSITY + NUM_STATES * indexR];
		real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR] / densityR;
		
		interfaceVelocityBuffer[side + DIM * index] = .5f * (velocityL + velocityR);
	}
}

__kernel void calcVelocityFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* interfaceVelocityBuffer,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;
	
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	int indexR = index;

	for (int side = 0; side < DIM; ++side) {	
		int indexL = index - stepsize[side];
		int indexL2 = indexL - stepsize[side];
		int indexR2 = index + stepsize[side];

		real interfaceVelocity = interfaceVelocityBuffer[side + DIM * index];
		//real theta = step(0.f, interfaceVelocity);
	
		for (int j = 0; j < NUM_STATES; ++j) {
			real stateR2 = stateBuffer[j + NUM_STATES * indexR2];
			real stateR = stateBuffer[j + NUM_STATES * indexR];
			real stateL = stateBuffer[j + NUM_STATES * indexL];
			real stateL2 = stateBuffer[j + NUM_STATES * indexL2];
			
			real deltaStateL = stateL - stateL2;
			real deltaState = stateR - stateL;
			real deltaStateR = stateR2 - stateR;
			
			//3D case crashes?
			//real flux = mix(stateR, stateL, theta) * interfaceVelocity;

			//this line crashes when compiling on my Intel HD4000 only for the 3D case
			//real stateSlopeRatio = mix(deltaStateR, deltaStateL, theta) / deltaState;
			//...but writing it out explicitly works fine
			real stateSlopeRatio;
			real flux;
			if (interfaceVelocity >= 0.f) {
				stateSlopeRatio = deltaStateL / deltaState;
				flux = stateL * interfaceVelocity;
			} else {
				stateSlopeRatio = deltaStateR / deltaState;
				flux = stateR * interfaceVelocity;
			}

			//2nd order
			real phi = slopeLimiter(stateSlopeRatio);
			real delta = phi * deltaState;
			flux += delta * .5f * fabs(interfaceVelocity) * (1.f - fabs(interfaceVelocity * dt_dx[side])) / (float)DIM;
			
			fluxBuffer[j + NUM_STATES * (side + DIM * index)] = flux;
		}
	}
}

__kernel void calcInterfaceMagneticField(
	__global real* interfaceMagneticFieldBuffer,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1 
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	int indexR = index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		
		real magneticFieldL = stateBuffer[side+STATE_MAGNETIC_FIELD_X + NUM_STATES * indexL];
		real magneticFieldR = stateBuffer[side+STATE_MAGNETIC_FIELD_X + NUM_STATES * indexR];
		
		interfaceMagneticFieldBuffer[side + DIM * index] = .5f * (magneticFieldL + magneticFieldR);
	}
}

__kernel void calcMagneticFieldFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* interfaceMagneticFieldBuffer,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;
	
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	int indexR = index;

	for (int side = 0; side < DIM; ++side) {	
		int indexL = index - stepsize[side];
		int indexL2 = indexL - stepsize[side];
		int indexR2 = index + stepsize[side];

		real interfaceMagneticField = interfaceMagneticFieldBuffer[side + DIM * index];
		//real theta = step(0.f, interfaceMagneticField);
	
		for (int j = 0; j < NUM_STATES; ++j) {
			real stateR2 = stateBuffer[j + NUM_STATES * indexR2];
			real stateR = stateBuffer[j + NUM_STATES * indexR];
			real stateL = stateBuffer[j + NUM_STATES * indexL];
			real stateL2 = stateBuffer[j + NUM_STATES * indexL2];
			
			real deltaStateL = stateL - stateL2;
			real deltaState = stateR - stateL;
			real deltaStateR = stateR2 - stateR;
			
			//3D case crashes?
			//real flux = mix(stateR, stateL, theta) * interfaceMagneticField;

			//this line crashes when compiling on my Intel HD4000 only for the 3D case
			//real stateSlopeRatio = mix(deltaStateR, deltaStateL, theta) / deltaState;
			//...but writing it out explicitly works fine
			real stateSlopeRatio;
			real flux;
			if (interfaceMagneticField >= 0.f) {
				stateSlopeRatio = deltaStateL / deltaState;
				flux = stateL * interfaceMagneticField;
			} else {
				stateSlopeRatio = deltaStateR / deltaState;
				flux = stateR * interfaceMagneticField;
			}

			//2nd order
			real phi = slopeLimiter(stateSlopeRatio);
			real delta = phi * deltaState;
			flux += delta * .5f * fabs(interfaceMagneticField) * (1.f - fabs(interfaceMagneticField * dt_dx[side])) / (float)DIM;
			
			fluxBuffer[j + NUM_STATES * (side + DIM * index)] = flux;
		}
	}
}

//the rest matches Burgers
//... except 1/2 B^2 is added to pressure

__kernel void calcFluxDeriv(
	__global real* derivBuffer,	//dstate/dt
	const __global real* fluxBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexNext = index + stepsize[side];
		for (int j = 0; j < NUM_STATES; ++j) {
			real fluxL = fluxBuffer[j + NUM_STATES * (side + DIM * index)];
			real fluxR = fluxBuffer[j + NUM_STATES * (side + DIM * indexNext)];
			real deltaFlux = fluxR - fluxL;
			deriv[j] -= deltaFlux / dx[side];
		}
	}
}

__kernel void computePressure(
	__global real* pressureBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 1 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 1 || i.y >= SIZE_Y - 1 
#if DIM > 2
		|| i.z < 1 || i.z >= SIZE_Z - 1
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	const __global real* state = stateBuffer + NUM_STATES * index;

	real density = state[STATE_DENSITY];
	real energyTotal = state[STATE_ENERGY_TOTAL];
	real4 velocity = (real4)(state[STATE_MOMENTUM_X], state[STATE_MOMENTUM_Y], state[STATE_MOMENTUM_Z], 0.f) / density;
	real4 magneticField = (real4)(state[STATE_MAGNETIC_FIELD_X], state[STATE_MAGNETIC_FIELD_Y], state[STATE_MAGNETIC_FIELD_Z], 0.f);
	
	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5f * dot(velocity, velocity);
	real specificEnergyMagnetic = .5f * dot(magneticField, magneticField);
	real specificEnergyPotential = potentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential - specificEnergyMagnetic;
	real pressure = (gamma - 1.f) * density * specificEnergyInternal + specificEnergyMagnetic;
	//von Neumann-Richtmyer artificial viscosity
	real deltaVelocitySq = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];	

		real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL];
		real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR];
		const float ZETA = 2.f;
		real deltaVelocity = ZETA * .5f * (velocityR - velocityL);
		deltaVelocitySq += deltaVelocity * deltaVelocity; 
	}
	pressure += deltaVelocitySq * density;
	pressureBuffer[index] = pressure;
}

__kernel void diffuseMomentum(
	__global real* derivBuffer,
	const __global real* pressureBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];	

		real pressureL = pressureBuffer[indexL];
		real pressureR = pressureBuffer[indexR];

		real deltaPressure = .5f * (pressureR - pressureL);
		deriv[side + STATE_MOMENTUM_X] -= deltaPressure / dx[side];
	}
}

__kernel void diffuseWork(
	__global real* derivBuffer,
	const __global real* stateBuffer,
	const __global real* pressureBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];

		real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL] / stateBuffer[STATE_DENSITY + NUM_STATES * indexL];
		real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR] / stateBuffer[STATE_DENSITY + NUM_STATES * indexR];
		
		real pressureL = pressureBuffer[indexL];
		real pressureR = pressureBuffer[indexR];

		real deltaWork = .5f * (pressureR * velocityR - pressureL * velocityL);

		deriv[STATE_ENERGY_TOTAL] -= deltaWork / dx[side];
	}
}

__kernel void calcMagneticFieldDivergence(
	__global real* magneticFieldDivergenceBuffer,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	real divergence = (stateBuffer[STATE_MAGNETIC_FIELD_X + NUM_STATES * (index + stepsize.x)]
		- stateBuffer[STATE_MAGNETIC_FIELD_X + NUM_STATES * (index - stepsize.x)]) / (2.f * DX);
#if DIM > 1
	divergence += (stateBuffer[STATE_MAGNETIC_FIELD_Y + NUM_STATES * (index + stepsize.y)]
		- stateBuffer[STATE_MAGNETIC_FIELD_Y + NUM_STATES * (index - stepsize.y)]) / (2.f * DY);
#if DIM > 2
	divergence += (stateBuffer[STATE_MAGNETIC_FIELD_Z + NUM_STATES * (index + stepsize.z)]
		- stateBuffer[STATE_MAGNETIC_FIELD_Z + NUM_STATES * (index - stepsize.z)]) / (2.f * DZ);
#endif
#endif

	magneticFieldDivergenceBuffer[index] = divergence;
}

__kernel void magneticPotentialPoissonRelax(
	__global real* magneticFieldPotentialBuffer,
	const __global real* magneticFieldDivergenceBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	real sum = (magneticFieldPotentialBuffer[index + stepsize.x] + magneticFieldPotentialBuffer[index - stepsize.x]) / (DX * DX);
#if DIM > 1
	sum += (magneticFieldPotentialBuffer[index + stepsize.y] + magneticFieldPotentialBuffer[index - stepsize.y]) / (DY * DY);
#if DIM > 2
	sum += (magneticFieldPotentialBuffer[index + stepsize.z] + magneticFieldPotentialBuffer[index - stepsize.z]) / (DZ * DZ);
#endif
#endif

	const real denom = -2.f * (1.f / (DX * DX)
#if DIM > 1
		+ 1.f / (DY * DY)
#endif
#if DIM > 2
		+ 1.f / (DZ * DZ)
#endif
	);

	//TODO double-buffer?
	magneticFieldPotentialBuffer[index] = (magneticFieldDivergenceBuffer[index] - sum) / denom;
}

__kernel void magneticFieldRemoveDivergence(
	__global real* stateBuffer,
	const __global real* magneticFieldPotentialBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	stateBuffer[STATE_MAGNETIC_FIELD_X + NUM_STATES * index] -= (magneticFieldPotentialBuffer[index + stepsize.x] - magneticFieldPotentialBuffer[index - stepsize.x]) / (2.f * DX);
#if DIM > 1
	stateBuffer[STATE_MAGNETIC_FIELD_Y + NUM_STATES * index] -= (magneticFieldPotentialBuffer[index + stepsize.y] - magneticFieldPotentialBuffer[index - stepsize.y]) / (2.f * DY);
#if DIM > 2
	stateBuffer[STATE_MAGNETIC_FIELD_Z + NUM_STATES * index] -= (magneticFieldPotentialBuffer[index + stepsize.z] - magneticFieldPotentialBuffer[index - stepsize.z]) / (2.f * DZ);
#endif
#endif
}


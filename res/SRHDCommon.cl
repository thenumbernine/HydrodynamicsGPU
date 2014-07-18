#include "HydroGPU/Shared/Common.h"

//velocity
#if DIM == 1
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_DENSITY_X], 0.f, 0.f, 0.f) / (ptr)[STATE_DENSITY])
#elif DIM == 2
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_DENSITY_X], (ptr)[STATE_MOMENTUM_DENSITY_Y], 0.f, 0.f) / (ptr)[STATE_DENSITY])
#elif DIM == 3
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_DENSITY_X], (ptr)[STATE_MOMENTUM_DENSITY_Y], (ptr)[STATE_MOMENTUM_DENSITY_Z], 0.f) / (ptr)[STATE_DENSITY])
#endif

/*
Incoming is Newtonian Euler equation state variables: density, momentum, newtonian total energy density
Outgoing is the SRHD primitives associated with it: density, velocity?, and pressure
		and the SRHD state variables:
*/
__kernel void initVariables(
	__global real* stateBuffer,
	__global real* primitiveBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	__global real* state = stateBuffer + NUM_STATES * index;
	__global real* primitive = primitiveBuffer + NUM_STATES * index;
/*
special modification for SRHD
 until I put more thought into the issue of unifying all iniital state variables for all solvers
I could go back to labelling initial states
then change the problems to provide either internal specific energy or newtonian pressure
and change the solvers C++ code to apply these transformations...
or I could just provide separate initial states for all Euler/MDH and all SRHD equations ...
or I could provide a wrapper like this ...
*/
	// calculate newtonian primitives from state vector
	real properRestMassDensity = state[0];
	real4 newtonianVelocity = (real4)(0.f, 0.f, 0.f, 0.f);
	newtonianVelocity.x = state[1] / state[0];
#if DIM > 1
	newtonianVelocity.y = state[2] / state[0];
#if DIM > 2
	newtonianVelocity.z = state[3] / state[0];
#endif
#endif
	real newtonianTotalEnergyDensity = state[DIM+1];
	real newtonianVelocitySq = dot(newtonianVelocity, newtonianVelocity);	
	real newtonianKineticSpecificEnergy = .5f * newtonianVelocitySq;	//eKin
	real newtonianTotalSpecificEnergy = newtonianTotalEnergyDensity / properRestMassDensity;	//eTot
	real internalSpecificEnergy = newtonianTotalSpecificEnergy - newtonianKineticSpecificEnergy;	//e
	// recast them as SR state variables 
	real pressure = (gamma - 1.f) * properRestMassDensity * internalSpecificEnergy;	//P
	real internalSpecificEnthalpy = 1.f + internalSpecificEnergy + pressure / properRestMassDensity; 	//h
	real lorentzFactor = 1.f / sqrt(1.f - newtonianVelocitySq);	//W = u0, ui = vi * u0
	real lorentzFactorSq = lorentzFactor * lorentzFactor;
	real restMassDensity = properRestMassDensity * lorentzFactor;	//D
	real4 momentumDensity = properRestMassDensity * internalSpecificEnthalpy * lorentzFactorSq * newtonianVelocity;	//S
	real totalEnergyDensity = properRestMassDensity * internalSpecificEnthalpy * lorentzFactorSq - pressure - restMassDensity;	//tau
	//write primitives
	primitive[PRIMITIVE_DENSITY] = properRestMassDensity;	//rho
	primitive[PRIMITIVE_VELOCITY_X] = newtonianVelocity.x;
#if DIM > 1
	primitive[PRIMITIVE_VELOCITY_Y] = newtonianVelocity.y;
#if DIM > 2
	primitive[PRIMITIVE_VELOCITY_Z] = newtonianVelocity.z;
#endif
#endif
	primitive[PRIMITIVE_PRESSURE] = pressure;
	//write state
	state[STATE_REST_MASS_DENSITY] = restMassDensity;	//D
	state[STATE_MOMENTUM_DENSITY_X] = momentumDensity.x;	//S1
#if DIM > 1
	state[STATE_MOMENTUM_DENSITY_Y] = momentumDensity.y;	//S2
#if DIM > 2
	state[STATE_MOMENTUM_DENSITY_Z] = momentumDensity.z;	//S3
#endif
#endif
	state[STATE_TOTAL_ENERGY_DENSITY] = totalEnergyDensity;	//tau
}

//specific to Euler equations
__kernel void convertToTex(
	const __global real* primitiveBuffer,
	const __global real* potentialBuffer,
	__write_only image3d_t fluidTex,
	__read_only image1d_t gradientTex,
	int displayMethod,
	float displayScale)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	const __global real* primitive = primitiveBuffer + NUM_STATES * index;

	real density = primitive[PRIMITIVE_DENSITY];
	real velocitySq = primitive[PRIMITIVE_VELOCITY_X] * primitive[PRIMITIVE_VELOCITY_X];
#if DIM > 1
	velocitySq += primitive[PRIMITIVE_VELOCITY_Y] * primitive[PRIMITIVE_VELOCITY_Y];
#endif
#if DIM > 2
	velocitySq += primitive[PRIMITIVE_VELOCITY_Z] * primitive[PRIMITIVE_VELOCITY_Z];
#endif
	real velocity = sqrt(velocitySq);
	real pressure = primitive[PRIMITIVE_PRESSURE];

#if DIM == 1
	float4 color = (float4)(density, velocity, pressure, 0.f) * displayScale;
#else
	real value;
	switch (displayMethod) {
	case DISPLAY_DENSITY:	//density
		value = density;
		break;
	case DISPLAY_VELOCITY:	//velocity
		value = velocity;
		break;
	case DISPLAY_PRESSURE:	//pressure
		value = pressure;
		break;
	case DISPLAY_POTENTIAL:
		value = potentialBuffer[index];
		break;
	default:
		value = .5f;
		break;
	}
	value *= displayScale;

	float4 color = read_imagef(gradientTex, CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, value).bgra;
#endif
	write_imagef(fluidTex, (int4)(i.x, i.y, i.z, 0), color);
}

__kernel void poissonRelax(
	__global real* potentialBuffer,
	const __global real* stateBuffer,
	int4 repeat)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	real sum = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int4 iprev = i;
		int4 inext = i;
		if (repeat[side]) {
			iprev[side] = (iprev[side] + size[side] - 1) % size[side];
			inext[side] = (inext[side] + 1) % size[side];
		} else {
			iprev[side] = max(iprev[side] - 1, 0);
			inext[side] = min(inext[side] + 1, size[side] - 1);
		}
		int indexPrev = INDEXV(iprev);
		int indexNext = INDEXV(inext);
		sum += potentialBuffer[indexPrev] + potentialBuffer[indexNext];
	}
	
#define M_PI 3.141592653589793115997963468544185161590576171875f
	real scale = M_PI * GRAVITATIONAL_CONSTANT * DX;
#if DIM > 1
	scale *= DY; 
#endif
#if DIM > 2
	scale *= DZ; 
#endif
	real density = stateBuffer[STATE_REST_MASS_DENSITY + NUM_STATES * index];
	potentialBuffer[index] = sum / (2.f * (float)DIM) + scale * density;
}

//TODO FIXME
__kernel void addGravity(
	__global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;

	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	real density = stateBuffer[STATE_REST_MASS_DENSITY + NUM_STATES * index];

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];
	
		real gravityGrad = .5f * (potentialBuffer[indexR] - potentialBuffer[indexL]);
		
		stateBuffer[side+STATE_MOMENTUM_DENSITY_X + NUM_STATES * index] -= dt_dx[side] * density * gravityGrad;
		stateBuffer[STATE_TOTAL_ENERGY_DENSITY + NUM_STATES * index] -= dt * density * gravityGrad * stateBuffer[side+STATE_MOMENTUM_DENSITY_X + NUM_STATES * index];
	}
}


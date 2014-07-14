#include "HydroGPU/Shared/Common.h"

//velocity
#if DIM == 1
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_X], 0.f, 0.f, 0.f) / (ptr)[STATE_DENSITY])
#elif DIM == 2
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_X], (ptr)[STATE_MOMENTUM_Y], 0.f, 0.f) / (ptr)[STATE_DENSITY])
#elif DIM == 3
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_X], (ptr)[STATE_MOMENTUM_Y], (ptr)[STATE_MOMENTUM_Z], 0.f) / (ptr)[STATE_DENSITY])
#endif

//specific to Euler equations
__kernel void convertToTex(
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	__write_only image3d_t fluidTex,
	__read_only image1d_t gradientTex,
	int displayMethod,
	float displayScale)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	const __global real* state = stateBuffer + NUM_STATES * index;

	real density = state[STATE_DENSITY];
	real energyTotal = state[STATE_ENERGY_TOTAL];
	real velocitySq = state[STATE_MOMENTUM_X] * state[STATE_MOMENTUM_X];
#if DIM > 1
	velocitySq += state[STATE_MOMENTUM_Y] * state[STATE_MOMENTUM_Y];
#endif
#if DIM > 2
	velocitySq += state[STATE_MOMENTUM_Z] * state[STATE_MOMENTUM_Z];
#endif
	velocitySq /= density * density;
	real velocity = sqrt(velocitySq);
	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5f * velocitySq;
	real specificEnergyPotential = gravityPotentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;

#if DIM == 1
#if NUM_STATES == 8	//MHD
	real4 magneticField = (real4)(state[STATE_MAGNETIC_FIELD_X], state[STATE_MAGNETIC_FIELD_Y], state[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticFieldMagn = length(magneticField);
#else
	real magneticFieldMagn = 0.f;
#endif
	float4 color = (float4)(density, velocity, specificEnergyInternal, magneticFieldMagn) * displayScale;
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
		value = (gamma - 1.f) * specificEnergyInternal * density;
		break;
	case DISPLAY_GRAVITY_POTENTIAL:
		value = gravityPotentialBuffer[index];
		break;
#if NUM_STATES == 8	//MHD
	case DISPLAY_MAGNETIC_FIELD:
		value = magneticFieldMagn;
		break;
#endif
	default:
		value = .5f;
		break;
	}
	value *= displayScale;

	float4 color = read_imagef(gradientTex, CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, value).bgra;
#endif
	write_imagef(fluidTex, (int4)(i.x, i.y, i.z, 0), color);
}


constant float2 offset[6] = {
	(float2)(-.5f, 0.f),
	(float2)(.5f, 0.f),
	(float2)(.2f, .3f),
	(float2)(.5f, 0.f),
	(float2)(.2f, -.3f),
	(float2)(.5f, 0.f),
};

__kernel void createVelocityField(
	__global real* velocityFieldVertexBuffer,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	float scale)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int4 size = (int4)(get_global_size(0), get_global_size(1), get_global_size(2), 0);	
	int vertexIndex = i.x + size.x * (i.y + size.y * i.z);
	__global real* vertex = velocityFieldVertexBuffer + 6 * 3 * vertexIndex;
	
	float4 f = (float4)(
		((float)i.x + .5f) / (float)size.x,
		((float)i.y + .5f) / (float)size.y,
		((float)i.z + .5f) / (float)size.z,
		0.f);

	//times grid size divided by velocity field size
	float4 sf = (float4)(f.x * SIZE_X, f.y * SIZE_Y, f.z * SIZE_Z, 0.f);
	int4 si = (int4)(sf.x, sf.y, sf.z, 0);
	//float4 fp = (float4)(sf.x - (float)si.x, sf.y - (float)si.y, sf.z - (float)si.z, 0.f);
	
#if 1	//plotting velocity 
	int stateIndex = INDEXV(si);
	const __global real* state = stateBuffer + NUM_STATES * stateIndex;
	float4 velocity = VELOCITY(state);
#endif
#if 0	//plotting gravity
	int4 ixL = si; ixL.x = (ixL.x + SIZE_X - 1) % SIZE_X;
	int4 ixR = si; ixR.x = (ixR.x + 1) % SIZE_X;
	int4 iyL = si; iyL.y = (iyL.y + SIZE_X - 1) % SIZE_X;
	int4 iyR = si; iyR.y = (iyR.y + 1) % SIZE_X;
	//external force is negative the potential gradient
	float4 velocity = (float4)(
		gravityPotentialBuffer[INDEXV(ixL)] - gravityPotentialBuffer[INDEXV(ixR)],
		gravityPotentialBuffer[INDEXV(iyL)] - gravityPotentialBuffer[INDEXV(iyR)],
		0.f,
		0.f);
#endif

	//velocity is the first axis of the basis to draw the arrows
	//the second should be perpendicular to velocity
#if DIM < 3
	real4 tv = (real4)(-velocity.y, velocity.x, 0.f, 0.f);
#elif DIM == 3
	real4 vx = (real4)(0.f, -velocity.z, velocity.y, 0.f);
	real4 vy = (real4)(velocity.z, 0.f, -velocity.x, 0.f);
	real4 vz = (real4)(-velocity.y, velocity.x, 0.f, 0.f);
	real lxsq = dot(vx,vx);
	real lysq = dot(vy,vy);
	real lzsq = dot(vz,vz);
	real4 tv;
	if (lxsq > lysq) {	//x > y
		if (lxsq > lzsq) {	//x > z, x > y
			tv = vx;
		} else {	//z > x > y
			tv = vz;
		}
	} else {	//y >= x
		if (lysq > lzsq) {	//y >= x, y > z
			tv = vy;
		} else {	// z > y >= x
			tv = vz;
		}
	}
#endif

	for (int i = 0; i < 6; ++i) {
		vertex[0 + 3 * i] = f.x * (XMAX - XMIN) + XMIN + scale * (offset[i].x * velocity.x + offset[i].y * tv.x);
		vertex[1 + 3 * i] = f.y * (YMAX - YMIN) + YMIN + scale * (offset[i].x * velocity.y + offset[i].y * tv.y);
		vertex[2 + 3 * i] = f.z * (ZMAX - ZMIN) + ZMIN + scale * (offset[i].x * velocity.z + offset[i].y * tv.z);
	}
}

__kernel void poissonRelax(
	__global real* gravityPotentialBuffer,
	const __global real* stateBuffer)
{
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

	real sum = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int indexPrev = index - stepsize[side];
		int indexNext = index + stepsize[side];
		sum += gravityPotentialBuffer[indexPrev] + gravityPotentialBuffer[indexNext];
	}

	const real volumeElement = DX
#if DIM > 1
		* DY 
#endif
#if DIM > 2
		* DZ
#endif
	;

	//delta^2 Phi = 4 pi G rho
	const real pi = 3.141592653589793115997963468544185161590576171875f;
	const real G = GRAVITATIONAL_CONSTANT;		//6.67384e-11 m^3 / (kg s^2)
	const real fourPiGRho = 4.f * pi * G;
	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];
	gravityPotentialBuffer[index] = (fourPiGRho * density - sum / volumeElement) / (-2.f * (float)DIM / volumeElement);
}

__kernel void calcGravityDeriv(
	__global real* derivBuffer,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer)
{
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

	__global real* deriv = derivBuffer + NUM_STATES * index;
	const __global real* state = stateBuffer + NUM_STATES * index;

	for (int j = 0; j < NUM_STATES; ++j) {
		deriv[j] = 0.f;
	}

	real density = state[STATE_DENSITY];

	for (int side = 0; side < DIM; ++side) {
		int indexPrev = index - stepsize[side];
		int indexNext = index + stepsize[side];
	
		real gravityPotentialGradient = .5f * (gravityPotentialBuffer[indexNext] - gravityPotentialBuffer[indexPrev]);
	
		//gravitational force = -gradient of gravitational potential
		deriv[side + STATE_MOMENTUM_X] -= density * gravityPotentialGradient / dx[side];
		deriv[STATE_ENERGY_TOTAL] -= density * gravityPotentialGradient * state[side + STATE_MOMENTUM_X];
	}
}


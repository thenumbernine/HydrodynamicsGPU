#include "HydroGPU/Euler.h"

//specific to Euler equations
__kernel void convertToTex(
	__write_only image3d_t destTex,
	int displayMethod,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	const __global char* solidBuffer
#ifdef MHD
	, const __global real* magneticFieldDivergenceBuffer
#endif
	)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 0 || i.x >= SIZE_X
#if DIM > 1
		|| i.y < 0 || i.y >= SIZE_Y
#endif
#if DIM > 2
		|| i.z < 0 || i.z >= SIZE_Z
#endif
	) return;

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
	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5f * velocitySq;
	real specificEnergyPotential = gravityPotentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;

#ifdef MHD
	
	real4 magneticField = (real4)(state[STATE_MAGNETIC_FIELD_X], state[STATE_MAGNETIC_FIELD_Y], state[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticFieldMagn = length(magneticField);

#else	//!MHD
#endif

	real value;
	switch (displayMethod) {
	case DISPLAY_DENSITY:	//density
		value = density;
		break;
	case DISPLAY_VELOCITY:	//velocity
		value = sqrt(velocitySq);
		break;
	case DISPLAY_PRESSURE:	//pressure
		value = (gamma - 1.f) * specificEnergyInternal * density;
		break;
	case DISPLAY_POTENTIAL:
		value = gravityPotentialBuffer[index];
		break;
#ifdef MHD
	case DISPLAY_MAGNETIC_FIELD:
		value = magneticFieldMagn;
		break;
	case DISPLAY_MAGNETIC_DIVERGENCE:
		{

#if 1		//disable the magnetic divergence display.
			//TODO: libRocket, a gui, and some toggle buttons for each variable ...

value = 0; break;

#elif 0		//use the divergence buffer (which was used to explicitly remove divergence)
			
			value = magneticFieldDivergenceBuffer[index];

#elif 0		//manually calculate it again (which should be zero post-removed-divergence)

			value = 0.f;
			
			//debugging: show magnetic field divergence
			int4 ixp = i;
			ixp[0] = (ixp[0] + 1) % size[0];
			int4 ixn = i;
			ixn[0] = (ixn[0] + size[0] - 1) % size[0];
			value = dx[0] * (stateBuffer[STATE_MAGNETIC_FIELD_X + NUM_STATES * INDEXV(ixp)] - stateBuffer[STATE_MAGNETIC_FIELD_X + NUM_STATES * INDEXV(ixn)]);
#if DIM > 1
			int4 iyp = i;
			iyp[1] = (iyp[1] + 1) % size[1];
			int4 iyn = i;
			iyn[1] = (iyn[1] + size[1] - 1) % size[1];
			value += dx[1] * (stateBuffer[STATE_MAGNETIC_FIELD_Y + NUM_STATES * INDEXV(iyp)] - stateBuffer[STATE_MAGNETIC_FIELD_Y + NUM_STATES * INDEXV(iyn)]);
#endif
#if DIM > 2
			int4 izp = i;
			izp[2] = (izp[2] + 1) % size[2];
			int4 izn = i;
			izn[2] = (izn[2] + size[2] - 1) % size[2];
			value += dx[2] * (stateBuffer[STATE_MAGNETIC_FIELD_Z + NUM_STATES * INDEXV(izp)] - stateBuffer[STATE_MAGNETIC_FIELD_Z + NUM_STATES * INDEXV(izn)]);
#endif
#endif
		}
		break;
#endif
	default:
		value = .5f;
		break;
	}

	write_imagef(destTex, (int4)(i.x, i.y, i.z, 0), (float4)(value, 0.f, 0.f, 0.f));
}

constant float2 offset[6] = {
	(float2)(-.5f, 0.f),
	(float2)(.5f, 0.f),
	(float2)(.2f, .3f),
	(float2)(.5f, 0.f),
	(float2)(.2f, -.3f),
	(float2)(.5f, 0.f),
};

__kernel void updateVectorField(
	__global real* vectorFieldVertexBuffer,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	float scale)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int4 size = (int4)(get_global_size(0), get_global_size(1), get_global_size(2), 0);	
	int vertexIndex = i.x + size.x * (i.y + size.y * i.z);
	__global real* vertex = vectorFieldVertexBuffer + 6 * 3 * vertexIndex;
	
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
	real4 velocity = VELOCITY(state);
#endif
#if 0	//plotting gravity
	int4 ixL = si; ixL.x = (ixL.x + SIZE_X - 1) % SIZE_X;
	int4 ixR = si; ixR.x = (ixR.x + 1) % SIZE_X;
	int4 iyL = si; iyL.y = (iyL.y + SIZE_X - 1) % SIZE_X;
	int4 iyR = si; iyR.y = (iyR.y + 1) % SIZE_X;
	//external force is negative the potential gradient
	real4 velocity = (float4)(
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


#include "HydroGPU/Euler.h"

#define gamma idealGas_heatCapacityRatio	//laziness

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
	real specificEnergyKinetic = .5 * velocitySq;
	real specificEnergyPotential = gravityPotentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;

#ifdef MHD
	real4 magneticField = (real4)(state[STATE_MAGNETIC_FIELD_X], state[STATE_MAGNETIC_FIELD_Y], state[STATE_MAGNETIC_FIELD_Z], 0.);
	real magneticFieldMagn = length(magneticField);
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
		value = (gamma - 1.) * specificEnergyInternal * density;
		break;
	case DISPLAY_POTENTIAL:
		value = gravityPotentialBuffer[index];
		break;
#ifdef MHD
	case DISPLAY_MAGNETIC_FIELD:
		value = magneticFieldMagn;
		break;
	case DISPLAY_MAGNETIC_DIVERGENCE_BUFFER:
		value = magneticFieldDivergenceBuffer[index];
		break;
	case DISPLAY_MAGNETIC_DIVERGENCE_CALCULATED:
	case DISPLAY_MAGNETIC_DIVERGENCE_ERROR:
		{
			value = 0.;
			
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
		}
		
		if (displayMethod == DISPLAY_MAGNETIC_DIVERGENCE_ERROR) {
			value = fabs(value - magneticFieldDivergenceBuffer[index]);
		}
		
		break;
#endif	//MHD
	}

	write_imagef(destTex, (int4)(i.x, i.y, i.z, 0), (float4)(value, 0., 0., 0.));
}

constant float2 offset[6] = {
	(float2)(-.5, 0.),
	(float2)(.5, 0.),
	(float2)(.2, .3),
	(float2)(.5, 0.),
	(float2)(.2, -.3),
	(float2)(.5, 0.),
};

__kernel void updateVectorField(
	__global float* vectorFieldVertexBuffer,
	real scale,
	int displayMethod,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int4 size = (int4)(get_global_size(0), get_global_size(1), get_global_size(2), 0);	
	int vertexIndex = i.x + size.x * (i.y + size.y * i.z);
	__global float* vertex = vectorFieldVertexBuffer + 6 * 3 * vertexIndex;
	
	float4 f = (float4)(
		((float)i.x + .5) / (float)size.x,
		((float)i.y + .5) / (float)size.y,
		((float)i.z + .5) / (float)size.z,
		0.);

	//times grid size divided by field size
	float4 sf = (float4)(f.x * SIZE_X, f.y * SIZE_Y, f.z * SIZE_Z, 0.);
	int4 si = (int4)(sf.x, sf.y, sf.z, 0);
	//float4 fp = (float4)(sf.x - (float)si.x, sf.y - (float)si.y, sf.z - (float)si.z, 0.);
	
	int stateIndex = INDEXV(si);
	const __global real* state = stateBuffer + NUM_STATES * stateIndex;

	real4 field = (real4)(0., 0., 0., 0.);
	switch (displayMethod) {
	case VECTORFIELD_VELOCITY:
	case VECTORFIELD_MOMENTUM:
		field.x = state[STATE_MOMENTUM_X];
#if DIM > 1
		field.y = state[STATE_MOMENTUM_Y];
#endif
#if DIM > 2
		field.z = state[STATE_MOMENTUM_Z];
#endif
		if (displayMethod == VECTORFIELD_MOMENTUM) field *= 1. / state[STATE_DENSITY];
		break;
	case VECTORFIELD_GRAVITY:
		{
			real4 grad = (real4)(0., 0., 0., 0.);
			int4 ixL = si; ixL.x = (ixL.x + SIZE_X - 1) % SIZE_X;
			int4 ixR = si; ixR.x = (ixR.x + 1) % SIZE_X;
			grad.x = (gravityPotentialBuffer[INDEXV(ixR)] - gravityPotentialBuffer[INDEXV(ixL)]) / (2. * DX);
#if DIM > 1	
			int4 iyL = si; iyL.y = (iyL.y + SIZE_Y - 1) % SIZE_Y;
			int4 iyR = si; iyR.y = (iyR.y + 1) % SIZE_Y;
			grad.y = (gravityPotentialBuffer[INDEXV(iyR)] - gravityPotentialBuffer[INDEXV(iyL)]) / (2. * DY);
#endif
#if DIM > 2
			int4 izL = si; izL.z = (izL.z + SIZE_Z - 1) % SIZE_Z;
			int4 izR = si; izR.z = (izR.z + 1) % SIZE_Z;
			grad.z = (gravityPotentialBuffer[INDEXV(izR)] - gravityPotentialBuffer[INDEXV(izL)]) / (2. * DZ);
#endif
			//external force is negative the potential gradient
			field = -grad;
		}
		break;
#if DIM == 3
	case VECTORFIELD_VORTICITY:
		{
			int4 ixL = si; ixL.x = (ixL.x + SIZE_X - 1) % SIZE_X;
			real* stateXL = stateBuffer + NUM_STATES * INDEXV(ixL);
			int4 ixR = si; ixR.x = (ixR.x + 1) % SIZE_X;
			real* stateXR = stateBuffer + NUM_STATES * INDEXV(ixR);
			int4 iyL = si; iyL.y = (iyL.y + SIZE_Y - 1) % SIZE_Y;
			real* stateYL = stateBuffer + NUM_STATES * INDEXV(iyL);
			int4 iyR = si; iyR.y = (iyR.y + 1) % SIZE_Y;
			real* stateYR = stateBuffer + NUM_STATES * INDEXV(iyR);
			int4 izL = si; izL.z = (izL.z + SIZE_Z - 1) % SIZE_Z;
			real* stateZL = stateBuffer + NUM_STATES * INDEXV(izL);
			int4 izR = si; izR.z = (izR.z + 1) % SIZE_Z;
			real* stateZR = stateBuffer + NUM_STATES * INDEXV(izR);
			
			// d/dy velocity.z - d/dz velocity.y
			field.x = (stateYR[STATE_MOMENTUM_Z] / stateYR[STATE_DENSITY] - stateYL[STATE_MOMENTUM_Z] / stateYL[STATE_DENSITY]) / (2. * DX)
					- (stateZR[STATE_MOMENTUM_Y] / stateZR[STATE_DENSITY] - stateZL[STATE_MOMENTUM_Y] / stateZL[STATE_DENSITY]) / (2. * DX);
			// d/dz velocity.x - d/dx velocity.z
			field.y = (stateZR[STATE_MOMENTUM_X] / stateZR[STATE_DENSITY] - stateZL[STATE_MOMENTUM_X] / stateZL[STATE_DENSITY]) / (2. * DY)
					- (stateXR[STATE_MOMENTUM_Z] / stateXR[STATE_DENSITY] - stateXL[STATE_MOMENTUM_Z] / stateXL[STATE_DENSITY]) / (2. * DY);
			// d/dx velocity.y - d/dy velocity.x
			field.z = (stateXR[STATE_MOMENTUM_Y] / stateXR[STATE_DENSITY] - stateXL[STATE_MOMENTUM_Y] / stateXL[STATE_DENSITY]) / (2. * DZ)
					- (stateYR[STATE_MOMENTUM_X] / stateYR[STATE_DENSITY] - stateYL[STATE_MOMENTUM_X] / stateYL[STATE_DENSITY]) / (2. * DZ);
		}
		break;
#endif
	}

	//field is the first axis of the basis to draw the arrows
	//the second should be perpendicular to field
#if DIM < 3
	real4 tv = (real4)(-field.y, field.x, 0., 0.);
#elif DIM == 3
	real4 vx = (real4)(0., -field.z, field.y, 0.);
	real4 vy = (real4)(field.z, 0., -field.x, 0.);
	real4 vz = (real4)(-field.y, field.x, 0., 0.);
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
		vertex[0 + 3 * i] = f.x * (XMAX - XMIN) + XMIN + scale * (offset[i].x * field.x + offset[i].y * tv.x);
		vertex[1 + 3 * i] = f.y * (YMAX - YMIN) + YMIN + scale * (offset[i].x * field.y + offset[i].y * tv.y);
		vertex[2 + 3 * i] = f.z * (ZMAX - ZMIN) + ZMIN + scale * (offset[i].x * field.z + offset[i].y * tv.z);
	}
}

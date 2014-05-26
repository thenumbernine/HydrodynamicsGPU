#include "roe_cell.h"

__kernel void calcEigenDecomposition(
	__global Cell* cells,
	int2 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		__global Interface *interface = cell->interfaces + side;
		if (interface->solid) continue;
		
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		__global Cell *cellL = cells + indexPrev;
		__global Cell *cellR = cell;
		
		real2 normal = (real2)(0.f, 0.f);
		normal[side] = 1;

		real densityL = cellL->q[0];
		real2 velocityL = (real2)(cellL->q[1], cellL->q[2]) / densityL;
		real energyTotalL = cellL->q[3] / densityL;

		real densityR = cellR->q[0];
		real2 velocityR = (real2)(cellR->q[1], cellR->q[2]) / densityR;
		real energyTotalR = cellR->q[3] / densityR;

		real velocitySqL = dot(velocityL, velocityL);
		real energyKineticL = .5f * velocitySqL;
		real energyThermalL = energyTotalL - energyKineticL;
		real pressureL = (GAMMA - 1.f) * densityL * energyThermalL;
		//real speedOfSoundL = sqrt(GAMMA * pressureL / densityL);
		real enthalpyTotalL = energyTotalL + pressureL / densityL;
		real weightL = sqrt(densityL);

		real velocitySqR = dot(velocityR, velocityR);
		real energyKineticR = .5f * velocitySqR;
		real energyThermalR = energyTotalR - energyKineticR;
		real pressureR = (GAMMA - 1.f) * densityR * energyThermalR;
		//real speedOfSoundR = sqrt(GAMMA * pressureR / densityR);
		real enthalpyTotalR = energyTotalR + pressureR / densityR;
		real weightR = sqrt(densityR);

		real denom = weightL + weightR;
		real2 velocity = (weightL * velocityL + weightR * velocityR) / denom;
		real velocitySq = dot(velocity, velocity);
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) / denom;
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
		real2 tangent = (real2)(-normal.y, normal.x);
		real velocityN = dot(velocity, normal);
		real velocityT = dot(velocity, tangent);
	
		//eigenvalues

		interface->eigenvalues[0]  = velocityN - speedOfSound;
		interface->eigenvalues[1]  = velocityN;
		interface->eigenvalues[2]  = velocityN;
		interface->eigenvalues[3]  = velocityN - speedOfSound;

		//min col 
		interface->eigenvectors[0][0] = 1.f;
		interface->eigenvectors[1][0] = velocity.x - speedOfSound * normal.x;
		interface->eigenvectors[2][0] = velocity.y - speedOfSound * normal.y;
		interface->eigenvectors[3][0] = enthalpyTotal - speedOfSound * velocityN;
		//mid col (normal)
		interface->eigenvectors[0][1] = 1.f;
		interface->eigenvectors[1][1] = velocity.x;
		interface->eigenvectors[2][1] = velocity.y;
		interface->eigenvectors[3][1] = .5f * velocitySq;
		//mid col (tangent)
		interface->eigenvectors[0][2] = 0.f;
		interface->eigenvectors[1][2] = tangent.x;
		interface->eigenvectors[2][2] = tangent.y;
		interface->eigenvectors[3][2] = velocityT;
		//max col 
		interface->eigenvectors[0][3] = 1.f;
		interface->eigenvectors[1][3] = velocity.x + speedOfSound * normal.x;
		interface->eigenvectors[2][3] = velocity.y + speedOfSound * normal.y;
		interface->eigenvectors[3][3] = enthalpyTotal + speedOfSound * velocityN;
		
		//calculate eigenvector inverses ... 
		//min row
		real scalar = .5f / (speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[0][0] = (.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocityN) * scalar;
		interface->eigenvectorsInverse[0][1] = -(normal.x * speedOfSound + (GAMMA - 1.f) * velocity.x) * scalar;
		interface->eigenvectorsInverse[0][2] = -(normal.y * speedOfSound + (GAMMA - 1.f) * velocity.y) * scalar;
		interface->eigenvectorsInverse[0][3] = (GAMMA - 1.f) * scalar;
		//mid normal row
		interface->eigenvectorsInverse[1][0] = 1.f - (GAMMA - 1.f) * velocitySq * scalar;
		interface->eigenvectorsInverse[1][1] = (GAMMA - 1.f) * velocity.x * 2.f * scalar;
		interface->eigenvectorsInverse[1][2] = (GAMMA - 1.f) * velocity.y * 2.f * scalar;
		interface->eigenvectorsInverse[1][3] = -(GAMMA - 1.f) * 2.f * scalar;
		//mid tangent row
		interface->eigenvectorsInverse[2][0] = -velocityT; 
		interface->eigenvectorsInverse[2][1] = tangent.x;
		interface->eigenvectorsInverse[2][2] = tangent.y;
		interface->eigenvectorsInverse[2][3] = 0.f;
		//max row
		interface->eigenvectorsInverse[3][0] = (.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocityN) * scalar;
		interface->eigenvectorsInverse[3][1] = (normal.x * speedOfSound - (GAMMA - 1.f) * velocity.x) * scalar;
		interface->eigenvectorsInverse[3][2] = (normal.y * speedOfSound - (GAMMA - 1.f) * velocity.y) * scalar;
		interface->eigenvectorsInverse[3][3] = (GAMMA - 1.f) * scalar;
	}
}

__kernel void calcDeltaQTilde(
	__global Cell* cells,
	int2 size) 
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		__global Interface *interface = cell->interfaces + side;
		
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		__global Cell *cellL = cells + indexPrev;
		__global Cell *cellR = cell;

		real4 deltaQ = cellR->q - cellL->q;
		//multiply by inverse
		for (int j = 0; j < 4; ++j) {
			real sum = 0.f;
			for (int k = 0; k < 4; ++k) {
				sum += interface->eigenvectorsInverse[j][k] * deltaQ[k];
			}
			interface->deltaQTilde[j] = sum;
		}
		//interface->deltaQTilde = (real4)( 
		//	dot(interface->eigenvectorsInverse[0], deltaQ),
		//	dot(interface->eigenvectorsInverse[1], deltaQ),
		//	dot(interface->eigenvectorsInverse[2], deltaQ),
		//	dot(interface->eigenvectorsInverse[3], deltaQ));
	}
}

__kernel void calcRTilde(
	__global Cell* cells,
	int2 size) 
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;
		
		__global Interface *interfaceL = &cells[indexPrev].interfaces[side];
		__global Interface *interface = &cells[index].interfaces[side];
		__global Interface *interfaceR = &cells[indexNext].interfaces[side];

		for (int state = 0; state < 4; ++state) {
			if (fabs(interface->deltaQTilde[state]) > 0.f) {
				if (interface->eigenvalues[state] > 0.f) {
					interface->rTilde[state] = interfaceL->deltaQTilde[state] / interface->deltaQTilde[state];
				} else {
					interface->rTilde[state] = interfaceR->deltaQTilde[state] / interface->deltaQTilde[state];
				}
			} else {
				interface->rTilde[state] = 0.f;
			}
		}
	}
}

constant real4 zero4 = (real4)(0.f, 0.f, 0.f, 0.f);
constant real4 one4 = (real4)(1.f, 1.f, 1.f, 1.f);
constant real4 two4 = (real4)(2.f, 2.f, 2.f, 2.f);
real4 fluxMethod(real4 r);
real4 fluxMethod(real4 r) {
	//superbee
	return max(zero4, max(min(one4, 2.f * r), min(two4, r)));
}

__kernel void calcFlux(
	__global Cell* cells,
	int2 size,
	real2 dt_dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		__global Cell *cellL = cells + indexPrev;
		__global Cell *cellR = cell;
		__global Interface *interface = cell->interfaces + side;

		real4 qAvg = (cellR->q + cellL->q) * .5f;
		
		//real4 fluxAvgTilde = (real4)(
		//	dot(interface->eigenvectorsInverse[0], qAvg),
		//	dot(interface->eigenvectorsInverse[1], qAvg),
		//	dot(interface->eigenvectorsInverse[2], qAvg),
		//	dot(interface->eigenvectorsInverse[3], qAvg)) * interface->eigenvalues;
		real4 fluxAvgTilde;
		for (int j = 0; j < 4; ++j) {
			real sum = 0.f;
			for (int k = 0; k < 4; ++k) {
				sum += interface->eigenvectorsInverse[j][k] * qAvg[k];
			}
			fluxAvgTilde[j] = sum * interface->eigenvalues[j];
		}
		
		fluxAvgTilde = fluxAvgTilde * interface->eigenvalues;
	
		real4 phi = fluxMethod(interface->rTilde);
		real4 theta = step(zero4, interface->eigenvalues) * 2.f - one4;
		real4 epsilon = interface->eigenvalues * dt_dx[side];
		real4 deltaFluxTilde = interface->eigenvalues * interface->deltaQTilde;
		real4 fluxTilde = fluxAvgTilde - .5f * deltaFluxTilde * (theta + phi * (epsilon - theta));
		
		//interface->flux = (real4)(
		//	dot(interface->eigenvectors[0], fluxTilde),
		//	dot(interface->eigenvectors[1], fluxTilde),
		//	dot(interface->eigenvectors[2], fluxTilde),
		//	dot(interface->eigenvectors[3], fluxTilde));
		for (int j = 0; j < 4; ++j) {
			real sum = 0.f;
			for (int k = 0; k < 4; ++k) {
				sum += interface->eigenvectors[j][k] * fluxTilde[k];
			}
			interface->flux[j] = sum;
		}
	}
}

__kernel void updateState(
	__global Cell* cells,
	int2 size,
	real2 dt_dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;
		
		__global Interface *interfaceL = &cells[index].interfaces[side];
		__global Interface *interfaceR = &cells[indexNext].interfaces[side];

		real4 df = interfaceR->flux - interfaceL->flux;
		cell->q -= df * dt_dx[side];
	}
}

__kernel void convertToTex(
	__global Cell* cells,
	int2 size,
	__write_only image2d_t fluidTex,
	__read_only image2d_t gradientTex)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;

#if 0	//cell position
	float2 f = (float2)(i.x, i.y) / (float2)(size.x, size.y);
	float4 color = (float4)(
		fabs(cell->x.x - (f.x - .5f)),
		fabs(cell->x.y - (f.y - .5f)),
		0.f, 1.f);
	write_imagef(fluidTex, i, color);
#endif
#if 0	//plot eigenbasis error 
	__global Interface *interface = &cell->interfaces[0];
	// a_ij = u_ik w_k v_kj
	// delta_ij = u_ik v_kj
	real4 check[4] = {
		(real4)(1.f, 0.f, 0.f, 0.f),
		(real4)(0.f, 1.f, 0.f, 0.f),
		(real4)(0.f, 0.f, 1.f, 0.f),
		(real4)(0.f, 0.f, 0.f, 1.f)
	};
	float err = 0.f;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			real sum = 0.f;
			for (int k = 0; k < 4; ++k) {
				sum += interface->eigenvectors[i][k] * interface->eigenvectorsInverse[k][j];
			}
			err += fabs(sum - check[i][j]);
		}
	}
	float4 color = (float4)(err, 0.f, 0.f, 1.f);
	write_imagef(fluidTex, i, color);
#endif
#if 0	//plot eigenstate reconstruction vs calculated flux
	__global Interface *interface = &cell->interfaces[0];
	// a_ij = u_ik w_k v_kj
	// delta_ij = u_ik v_kj
	real4 check[4] = {
		(real4)(0.f, 1.f, 0.f, 0.f),	//desires: 0 1 0 0, results: 0 0 0 -.5
		(real4)(0.f, 1.f, 0.f, 0.f),	//still haven't set up the rest of these vectors correctly ... would have to reconstruct the flux here, or store it up there ...
		(real4)(0.f, 0.f, 1.f, 0.f),	// ... or do the error detection up there and store the result
		(real4)(0.f, 0.f, 0.f, 1.f)
	};
	float err = 0.;
//	for (int i = 0; i < 4; ++i) {
//		for (int j = 0; j < 4; ++j) {
	{ { int i = 0; int j = 3;
			real sum = 0.f;
			for (int k = 0; k < 4; ++k) {
				sum += interface->eigenvectors[i][k] * interface->eigenvalues[k] * interface->eigenvectorsInverse[k][j];
			}
			err += fabs(sum - check[i][j]);
		}
	}
	float4 color = (float4)(err, 0.f, 0.f, 1.f);
	write_imagef(fluidTex, i, color);
#endif
#if 1	//plot density
	float4 color = read_imagef(gradientTex, 
		CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR,
		(float2)(cell->q[0] * 2.f, .5f));
	write_imagef(fluidTex, i, color.bgra);
#endif
}


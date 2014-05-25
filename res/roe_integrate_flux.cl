#include "roe_cell.h"

__kernel void calcEigenDecomposition(
	__global Cell* cells,
	int2 size)
{
	int2 i = int2(get_global_id(0), get_global_id(1));
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
		
		real2 normal = real2(0., 0.);
		normal[side] = 1;

		real densityL = cellL->q[0];
		real2 velocityL = real2(cellL->q[1], cellL->q[2]) / densityL;
		real energyTotalL = cellL->q[3] / densityL;

		real densityR = cellR->q[0];
		real2 velocityR = real2(cellR->q[1], cellR->q[2]) / densityR;
		real energyTotalR = cellR->q[3] / densityR;

		real velocitySqL = dot(velocityL, velocityL);
		real energyKineticL = .5 * velocitySqL;
		real energyThermalL = energyTotalL - energyKineticL;
		real pressureL = (GAMMA - 1.) * densityL * energyThermalL;
		real speedOfSoundL = sqrt(GAMMA * pressureL / densityL);
		real enthalpyTotalL = energyTotalL + pressureL / densityL;
		real weightL = sqrt(densityL);

		real velocitySqR = dot(velocityR, velocityR);
		real energyKineticR = .5 * velocitySqR;
		real energyThermalR = energyTotalR - energyKineticR;
		real pressureR = (GAMMA - 1.) * densityR * energyThermalR;
		real speedOfSoundR = sqrt(GAMMA * pressureR / densityR);
		real enthalpyTotalR = energyTotalR + pressureR / densityR;
		real weightR = sqrt(densityR);

		real denom = weightL + weightR;
		real2 velocity = (weightL * velocityL + weightR * velocityR) / denom;
		real velocitySq = dot(velocity, velocity);
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) / denom;
		real speedOfSound = sqrt((GAMMA - 1.) * (enthalpyTotal - .5 * velocitySq));
		real2 tangent = real2(-normal.y, normal.x);
		real velocityN = dot(velocity, normal);
		real velocityT = dot(velocity, tangent);
	
		//eigenvalues

		interface->eigenvalues[0]  = velocityN - speedOfSound;
		interface->eigenvalues[1]  = velocityN;
		interface->eigenvalues[2]  = velocityN;
		interface->eigenvalues[3]  = velocityN - speedOfSound;

		//min col 
		interface->eigenvectors[0][0] = 1.;
		interface->eigenvectors[0][1] = velocity.x - speedOfSound * normal.x;
		interface->eigenvectors[0][2] = velocity.y - speedOfSound * normal.y;
		interface->eigenvectors[0][3] = enthalpyTotal - speedOfSound * velocityN;
		//mid col (normal)
		interface->eigenvectors[1][0] = 1.;
		interface->eigenvectors[1][1] = velocity.x;
		interface->eigenvectors[1][2] = velocity.y;
		interface->eigenvectors[1][3] = .5 * velocitySq;
		//mid col (tangent)
		interface->eigenvectors[2][0] = 0.;
		interface->eigenvectors[2][1] = tangent.x;
		interface->eigenvectors[2][2] = tangent.y;
		interface->eigenvectors[2][3] = velocityT;
		//max col 
		interface->eigenvectors[3][0] = 1.;
		interface->eigenvectors[3][1] = velocity.x + speedOfSound * normal.x;
		interface->eigenvectors[3][2] = velocity.y + speedOfSound * normal.y;
		interface->eigenvectors[3][3] = enthalpyTotal + speedOfSound * velocityN;
		
		//calculate eigenvector inverses ... 
		//min row
		interface->eigenvectorsInverse[0][0] = (.5 * (GAMMA - 1.) * velocitySq + speedOfSound * velocityN) / (2. * speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[1][0] = -(normal.x * speedOfSound + (GAMMA - 1.) * velocity.x) / (2. * speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[2][0] = -(normal.y * speedOfSound + (GAMMA - 1.) * velocity.y) / (2. * speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[3][0] = (GAMMA - 1.) / (2. * speedOfSound * speedOfSound);
		//mid normal row
		interface->eigenvectorsInverse[0][1] = 1. - .5 * (GAMMA - 1.) * velocitySq / (speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[1][1] = (GAMMA - 1.) * velocity.x / (speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[2][1] = (GAMMA - 1.) * velocity.y / (speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[3][1] = -(GAMMA - 1.) / (speedOfSound * speedOfSound);
		//mid tangent row
		interface->eigenvectorsInverse[0][2] = -velocityT; 
		interface->eigenvectorsInverse[1][2] = tangent.x;
		interface->eigenvectorsInverse[2][2] = tangent.y;
		interface->eigenvectorsInverse[3][2] = 0.;
		//max row
		interface->eigenvectorsInverse[0][3] = (.5 * (GAMMA - 1.) * velocitySq - speedOfSound * velocityN) / (2. * speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[1][3] = (normal.x * speedOfSound - (GAMMA - 1.) * velocity.x) / (2. * speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[2][3] = (normal.y * speedOfSound - (GAMMA - 1.) * velocity.y) / (2. * speedOfSound * speedOfSound);
		interface->eigenvectorsInverse[3][3] = (GAMMA - 1.) / (2. * speedOfSound * speedOfSound);
	}
}

__kernel void calcDeltaQTilde(
	__global Cell* cells,
	int2 size) 
{
	int2 i = int2(get_global_id(0), get_global_id(1));
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
		interface->deltaQTilde = 
			interface->eigenvectorsInverse[0] * deltaQ[0] +
			interface->eigenvectorsInverse[1] * deltaQ[1] +
			interface->eigenvectorsInverse[2] * deltaQ[2] +
			interface->eigenvectorsInverse[3] * deltaQ[3];
	}
}

__kernel void calcRTilde(
	__global Cell* cells,
	int2 size) 
{
	int2 i = int2(get_global_id(0), get_global_id(1));
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
			if (abs(interface->deltaQTilde[state] > 0.)) {
				if (interface->eigenvalues[state] > 0.) {
					interface->rTilde[state] = interfaceL->deltaQTilde[state] / interface->deltaQTilde[state];
				} else {
					interface->rTilde[state] = interfaceR->deltaQTilde[state] / interface->deltaQTilde[state];
				}
			} else {
				interface->rTilde[state] = 0.;
			}
		}
	}
}

constant real4 zero4 = real4(0., 0., 0., 0.);
constant real4 one4 = real4(1., 1., 1., 1.);
constant real4 two4 = real4(2., 2., 2., 2.);
real4 fluxMethod(real4 r) {
	//superbee
	return max(zero4, max(min(one4, 2. * r), min(two4, r)));
}

__kernel void calcFlux(
	__global Cell* cells,
	int2 size,
	real2 dt_dx)
{
	int2 i = int2(get_global_id(0), get_global_id(1));
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

		real4 qAvg = (cellR->q + cellL->q) * .5;
		real4 fluxAvgTilde =
			interface->eigenvectorsInverse[0] * qAvg[0] +
			interface->eigenvectorsInverse[1] * qAvg[1] +
			interface->eigenvectorsInverse[2] * qAvg[2] +
			interface->eigenvectorsInverse[3] * qAvg[3];
		fluxAvgTilde = fluxAvgTilde * interface->eigenvalues;
	
		real4 phi = fluxMethod(interface->rTilde);
		real4 theta = step(interface->eigenvalues, zero4) * 2. - one4;
		real4 epsilon = interface->eigenvalues * dt_dx[side];
		real4 deltaFluxTilde = interface->eigenvalues * interface->deltaQTilde;
		real4 fluxTilde = fluxAvgTilde - .5 * deltaFluxTilde * (theta + phi * (epsilon - theta));
		interface->flux = 
			interface->eigenvectors[0] * fluxTilde[0] +
			interface->eigenvectors[1] * fluxTilde[1] +
			interface->eigenvectors[2] * fluxTilde[2] +
			interface->eigenvectors[3] * fluxTilde[3];
	}
}

__kernel void updateState(
	__global Cell* cells,
	int2 size,
	real2 dt_dx)
{
	int2 i = int2(get_global_id(0), get_global_id(1));
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

__kernel void copyToTex(
	__global Cell* cells,
	int2 size,
	__global char *tex)
{
	int2 i = int2(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	__global char *pixel = tex + index;

	//map values here
	float value = (float)cell->q[0];
	int ivalue = (int)(value * 255.);
	*pixel = (char)ivalue;
}


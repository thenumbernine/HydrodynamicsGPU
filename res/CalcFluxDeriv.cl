/*
This function should only be used when the flux dimension equals the state dimension.
*OR*, if the flux is smaller, when the flux is only affecting the first variables of the state vectors.
In the case that the flux affects non-sequential state vectors, a custom function will have to be written.
For that reason, I should probably separate this into its own file, so other solvers can bypass it and overload it.
*/
__kernel void calcFluxDeriv(
	__global real* derivBuffer,
	const __global real* fluxBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif	//SOLID
)
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

#ifdef SOLID
	if (solidBuffer[index]) return;
#endif	//SOLID

	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int interfaceIndex = side + DIM * index;
		int interfaceIndexNext = interfaceIndex + DIM * stepsize[side];
		const __global real* fluxL = fluxBuffer + NUM_FLUX_STATES * interfaceIndex;
		const __global real* fluxR = fluxBuffer + NUM_FLUX_STATES * interfaceIndexNext;
		for (int j = 0; j < NUM_FLUX_STATES; ++j) {
			real deltaFlux = fluxR[j] - fluxL[j];
			deriv[j] -= deltaFlux / dx[side];
		}
	}
}

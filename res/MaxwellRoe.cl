/*
Hyperbolic formalism of Maxwell's equations
Described in Trangenstein, "Numeric Solutions of Hyperbolic Partial Differential Equations" Section 4.3
*/

#include "HydroGPU/Shared/Common.h"

__kernel void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	int side)
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

	int interfaceIndex = side + DIM * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
	__global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;

	//eigenvalues

	real eigenvalue = 1.f / (sqrtPermittivity * sqrtPermeability); 
	eigenvalues[0] = -eigenvalue;
	eigenvalues[1] = -eigenvalue;
	eigenvalues[2] = 0.f;
	eigenvalues[3] = 0.f;
	eigenvalues[4] = eigenvalue;
	eigenvalues[5] = eigenvalue;

	//eigenvectors

	const float M_SQRT_1_2 = 0.7071067811865475727373109293694142252206802368164f;
	
	//col
	eigenvectors[0 + NUM_STATES * 0] = 0.f;
	eigenvectors[1 + NUM_STATES * 0] = 0.f;
	eigenvectors[2 + NUM_STATES * 0] = sqrtPermittivity * M_SQRT_1_2;
	eigenvectors[3 + NUM_STATES * 0] = 0.f; 
	eigenvectors[4 + NUM_STATES * 0] = sqrtPermeability * M_SQRT_1_2;
	eigenvectors[5 + NUM_STATES * 0] = 0.f;
	//col
	eigenvectors[0 + NUM_STATES * 1] = 0.f;
	eigenvectors[1 + NUM_STATES * 1] = -sqrtPermittivity * M_SQRT_1_2;
	eigenvectors[2 + NUM_STATES * 1] = 0.f;
	eigenvectors[3 + NUM_STATES * 1] = 0.f;
	eigenvectors[4 + NUM_STATES * 1] = 0.f;
	eigenvectors[5 + NUM_STATES * 1] = sqrtPermeability * M_SQRT_1_2;
	//col
	eigenvectors[0 + NUM_STATES * 2] = -sqrtPermittivity * M_SQRT_1_2;
	eigenvectors[1 + NUM_STATES * 2] = 0.f;
	eigenvectors[2 + NUM_STATES * 2] = 0.f;
	eigenvectors[3 + NUM_STATES * 2] = sqrtPermeability * M_SQRT_1_2;
	eigenvectors[4 + NUM_STATES * 2] = 0.f;
	eigenvectors[5 + NUM_STATES * 2] = 0.f;
	//col
	eigenvectors[0 + NUM_STATES * 3] = sqrtPermittivity * M_SQRT_1_2;
	eigenvectors[1 + NUM_STATES * 3] = 0.f;
	eigenvectors[2 + NUM_STATES * 3] = 0.f;
	eigenvectors[3 + NUM_STATES * 3] = sqrtPermeability * M_SQRT_1_2;
	eigenvectors[4 + NUM_STATES * 3] = 0.f;
	eigenvectors[5 + NUM_STATES * 3] = 0.f;
	//col
	eigenvectors[0 + NUM_STATES * 4] = 0.f;
	eigenvectors[1 + NUM_STATES * 4] = sqrtPermittivity * M_SQRT_1_2;
	eigenvectors[2 + NUM_STATES * 4] = 0.f;
	eigenvectors[3 + NUM_STATES * 4] = 0.f;
	eigenvectors[4 + NUM_STATES * 4] = 0.f;
	eigenvectors[5 + NUM_STATES * 4] = sqrtPermeability * M_SQRT_1_2;
	//col
	eigenvectors[0 + NUM_STATES * 4] = 0.f;
	eigenvectors[1 + NUM_STATES * 4] = 0.f;
	eigenvectors[2 + NUM_STATES * 4] = -sqrtPermittivity * M_SQRT_1_2;
	eigenvectors[3 + NUM_STATES * 4] = 0.f;
	eigenvectors[4 + NUM_STATES * 4] = sqrtPermeability * M_SQRT_1_2;
	eigenvectors[5 + NUM_STATES * 4] = 0.f;

	//eigenvector inverses = 1/transpose of eigenvector
	for (int i = 0; i < NUM_STATES; ++i) {
		for (int j = 0; j < NUM_STATES; ++j) {
			eigenvectorsInverse[i + NUM_STATES * j] = 1.f / eigenvectors[j + NUM_STATES * i];
		}
	}

#if DIM > 1
	if (side == 1) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;

			//-90' rotation applied to the LHS of incoming velocity vectors, to move their y axis into the x axis
			// is equivalent of a -90' rotation applied to the RHS of the flux jacobian A
			// and A = Q V Q-1 for Q = the right eigenvectors and Q-1 the left eigenvectors
			// so a -90' rotation applied to the RHS of A is a +90' rotation applied to the RHS of Q-1 the left eigenvectors
			//and while a rotation applied to the LHS of a vector rotates the elements of its column vectors, a rotation applied to the RHS rotates the elements of its row vectors 
			//each row's y <- x, x <- -y
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_ELECTRIC_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_ELECTRIC_X] = eigenvectorsInverse[i + NUM_STATES * STATE_ELECTRIC_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_ELECTRIC_Y] = tmp;
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_Y] = tmp;
			//a -90' rotation applied to the RHS of A must be corrected with a 90' rotation on the LHS of A
			//this rotates the elements of the column vectors by 90'
			//each column's x <- y, y <- -x
			tmp = eigenvectors[STATE_ELECTRIC_X + NUM_STATES * i];
			eigenvectors[STATE_ELECTRIC_X + NUM_STATES * i] = eigenvectors[STATE_ELECTRIC_Y + NUM_STATES * i];
			eigenvectors[STATE_ELECTRIC_Y + NUM_STATES * i] = tmp;
			tmp = eigenvectors[STATE_MAGNETIC_X + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_X + NUM_STATES * i] = eigenvectors[STATE_MAGNETIC_Y + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_ELECTRIC_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_ELECTRIC_X] = eigenvectorsInverse[i + NUM_STATES * STATE_ELECTRIC_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_ELECTRIC_Z] = tmp;
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MAGNETIC_Z] = tmp;
			tmp = eigenvectors[STATE_ELECTRIC_X + NUM_STATES * i];
			eigenvectors[STATE_ELECTRIC_X + NUM_STATES * i] = eigenvectors[STATE_ELECTRIC_Z + NUM_STATES * i];
			eigenvectors[STATE_ELECTRIC_Z + NUM_STATES * i] = tmp;
			tmp = eigenvectors[STATE_MAGNETIC_X + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_X + NUM_STATES * i] = eigenvectors[STATE_MAGNETIC_Z + NUM_STATES * i];
			eigenvectors[STATE_MAGNETIC_Z + NUM_STATES * i] = tmp;
		}
	}
#endif
#endif
	
}

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
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
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 0);
#if DIM > 1
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 1);
#endif
#if DIM > 2
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 2);
#endif
}

__kernel void addSource(
	__global real* derivBuffer,
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
	
	const __global real* state = stateBuffer + NUM_STATES * index;
	real4 conductiveElectric = (real4)(state[STATE_ELECTRIC_X], state[STATE_ELECTRIC_Y], state[STATE_ELECTRIC_Z], 0.f) * (conductivity / permittivity);
	
	__global real* deriv = derivBuffer + NUM_STATES * index;
	deriv[STATE_ELECTRIC_X] -= conductiveElectric.x;
	deriv[STATE_ELECTRIC_Y] -= conductiveElectric.y;
	deriv[STATE_ELECTRIC_Z] -= conductiveElectric.z;
}


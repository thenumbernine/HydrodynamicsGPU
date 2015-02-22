#pragma once

enum {
	BOUNDARY_KERNEL_PERIODIC,
	BOUNDARY_KERNEL_MIRROR,		//	\_ combined to make up reflecting boundary conditions
	BOUNDARY_KERNEL_REFLECT,	//	/
	BOUNDARY_KERNEL_FREEFLOW,
	NUM_BOUNDARY_KERNELS
};


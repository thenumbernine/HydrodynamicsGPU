#pragma once

/*
Boundary methods are picked by the user
They map each different state to different boundary kernel operations
One example of the distinction comes from Mirror boundary conditions:
*/
enum {
	BOUNDARY_METHOD_NONE = -1,
	BOUNDARY_METHOD_PERIODIC,
	BOUNDARY_METHOD_MIRROR,
	BOUNDARY_METHOD_FREEFLOW,
	NUM_BOUNDARY_METHODS
};

//boundary kernels operate on the buffers themselves
enum {
	BOUNDARY_KERNEL_NONE = -1,
	BOUNDARY_KERNEL_PERIODIC,
	BOUNDARY_KERNEL_MIRROR,		//	\_ combined to make up reflecting boundary conditions
	BOUNDARY_KERNEL_REFLECT,	//	/
	BOUNDARY_KERNEL_FREEFLOW,
	NUM_BOUNDARY_KERNELS
};

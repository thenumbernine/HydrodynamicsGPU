#pragma once

#include "HydroGPU/Shared/Types.h"

struct Interface {
	//Roe-specific values
	real16 eigenvectors;			//stored row-major 
	real16 eigenvectorsInverse;	// so math matches array index notation
	real4 eigenvalues;
	real4 deltaQTilde;
	
	//base cell values
	real4 flux;
};
typedef struct Interface Interface;

struct Cell {
	//base cell values
	real4 q;

	Interface interfaces[DIM];
};
typedef struct Cell Cell;


#pragma once

//double depends on cl_khr_fp64 extension
// which isn't working on my machine ... 
typedef float real;

#define DIM	2
#define NUM_STATES	2+DIM

struct Interface {
	real x[DIM];
	real r[NUM_STATES];
	real flux[NUM_STATES];
	real velocity;
};
typedef struct Interface Interface;

struct Cell {
	real value;

	real state[NUM_STATES];
	real x[DIM];
	real pressure;

	Interface interfaces[DIM];
};
typedef struct Cell Cell;


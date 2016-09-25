#pragma once

#include "HydroGPU/Integrator/Integrator.h"
#include <array>

namespace HydroGPU {
namespace Integrator {

/*
ButcherTable template struct has specializations of <row, col> with a ::value() static method that returns the Butcher tableau a_ij coefficient, for row i and column j
The specializations of of row == order corresponds to the b_j coefficients, for column j
ButcherTableAlpha  and ButhcerTableBeta make up the coefficients used in 1.8 of 
http://www.ams.org/journals/mcom/1998-67-221/S0025-5718-98-00913-2/S0025-5718-98-00913-2.pdf
"TOTAL VARIATION DIMINISHING RUNGE-KUTTA SCHEMES" by SIGAL GOTTLIEB AND CHI-WANG SHU
*/
template<typename Tableau>
struct RungeKutta : public Integrator {
	enum { order = Tableau::Order };
	typedef Integrator Super;
	RungeKutta(HydroGPU::Solver::Solver* solver);
	virtual void integrate(real dt, std::function<void(cl::Buffer)> callback);
protected:

	std::array<cl::Buffer, order> stateBuffer;
	std::array<cl::Buffer, order> derivBuffer;
	cl::Kernel multAddKernel;
};

template<typename Tableau>
RungeKutta<Tableau>::RungeKutta(HydroGPU::Solver::Solver* solver) 
: Super(solver)
{
	int volume = solver->getVolume();

	for (int i = 0;  i < order; ++i) {
		bool needed = false;
		for (int m = i; m < order; ++m) {
			needed |= Tableau::alphas(m,i) != 0;
		}
		if (needed) {
			stateBuffer[i] = solver->cl.alloc(sizeof(real) * solver->numStates() * volume, std::string() + "RungeKutta::stateBuffer[" + std::to_string(i) + "]");
		}
		
		needed = false;
		for (int m = i; m < order; ++m) {
			needed |= Tableau::betas(m,i) != 0;
		}
		if (needed) {	
			derivBuffer[i] = solver->cl.alloc(sizeof(real) * solver->numStates() * volume, std::string() + "RungeKutta::derivBuffer[" + std::to_string(i) + "]");
		}
	}

	//put this in parent class of ForwardEuler and RungeKutta?
	multAddKernel = cl::Kernel(solver->program, "multAdd");
	multAddKernel.setArg(0, solver->stateBuffer);
	multAddKernel.setArg(1, solver->stateBuffer);
}

template<typename Tableau>
void RungeKutta<Tableau>::integrate(real dt, std::function<void(cl::Buffer)> callback) {
	size_t length = solver->numStates() * solver->getVolume();
	size_t bufferSize = sizeof(real) * length;
	cl::NDRange globalSize1d(length);

	//u(0) = u^n
	bool needed = false;
	for (int m = 0; m < order; ++m) {
		needed |= Tableau::alphas(m,0) != 0;
	}
	if (needed) {
		solver->commands.enqueueCopyBuffer(solver->stateBuffer, stateBuffer[0], 0, 0, bufferSize);
	}

	//L(u^(0))
	needed = false;
	for (int m = 0; m < order; ++m) {
		needed |= Tableau::betas(m,0) != 0;
	}
	if (needed) {
		solver->cl.zero(derivBuffer[0], bufferSize);
		callback(derivBuffer[0]);
	}
	
	for (int i = 1; i <= order; ++i) {
		//u^(i) = sum k=0 to i-1 of (alpha_ik u^(k) + dt beta_ik L(u^(k)) )
		solver->cl.zero(solver->stateBuffer, bufferSize);
		for (int k = 0; k < i; ++k) {
			if (Tableau::alphas(i-1,k)) {
				multAddKernel.setArg(2, stateBuffer[k]);
				multAddKernel.setArg(3, Tableau::alphas(i-1,k));
				solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
			}

			if (Tableau::betas(i-1,k)) {
				multAddKernel.setArg(2, derivBuffer[k]);
				multAddKernel.setArg(3, Tableau::betas(i-1,k) * dt);
				solver->commands.enqueueNDRangeKernel(multAddKernel, solver->offset1d, globalSize1d, solver->localSize1d);
			}
		}

		if (i < order) {
			//only do this if alpha_mi != 0 for any m
			//otherwise there's no need to store this buffer
			needed = false;
			for (int m = i; m < order; ++m) {
				needed |= Tableau::alphas(m,i) != 0;
			}	
			if (needed) {
				solver->commands.enqueueCopyBuffer(solver->stateBuffer, stateBuffer[i], 0, 0, bufferSize);
			}

			//likewise here, only if beta_mi != 0 for any m
			//with that in mind, no need to allocate these buffers unless they are needed.
			needed = false;
			for (int m = i; m < order; ++m) {
				needed |= Tableau::betas(m,i) != 0;
			}
			if (needed) {
				solver->cl.zero(derivBuffer[i], bufferSize);
				callback(derivBuffer[i]);
			}
		}
		//else just leave the state in there
	}
}

//the following are from https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Classic_fourth-order_method

struct RungeKutta2Tableau {
	enum { Order = 2 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0},
		{1, 0},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{.5, 0},
		{0, 1},
	}; return m[i][j]; }
};
typedef RungeKutta<RungeKutta2Tableau> RungeKutta2;

struct RungeKutta2HeunTableau {
	enum { Order = 2 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0},
		{1, 0},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{1, 0},
		{.5, .5},
	}; return m[i][j]; }
};
typedef RungeKutta<RungeKutta2HeunTableau> RungeKutta2Heun;

struct RungeKutta2RalstonTableau {
	enum { Order = 2 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0},
		{1, 0},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{2./3., 0},
		{1./4., 3./4.},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta2RalstonTableau> RungeKutta2Ralston;

struct RungeKutta3Tableau {
	enum { Order = 3 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0, 0},
		{1, 0, 0},
		{1, 0, 0},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{.5, 0, 0},
		{-1, 2, 0},
		{1./6., 2./6., 1./6.},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta3Tableau> RungeKutta3;

struct RungeKutta4Tableau {
	enum { Order = 4 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 0, 0},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{.5, 0, 0, 0},
		{0, .5, 0, 0},
		{0, 0, 1, 0},
		{1./6., 2./6., 2./6., 1./6.},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta4Tableau> RungeKutta4;

struct RungeKutta4_3_8thsRuleTableau {
	enum { Order = 4 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 0, 0},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{1./3., 0, 0, 0},
		{-1./3., 0, 0, 0},
		{1, -1, 1, 0},
		{1./8., 3./8., 3./8., 1./8.},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta4_3_8thsRuleTableau> RungeKutta4_3_8thsRule;

//the following are from http://www.ams.org/journals/mcom/1998-67-221/S0025-5718-98-00913-2/S0025-5718-98-00913-2.pdf

struct RungeKutta2TVDTableau {
	enum { Order = 2 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0},
		{.5, .5},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{1, 0},
		{0, .5},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta2TVDTableau> RungeKutta2TVD; 

struct RungeKutta2NonTVDTableau {
	enum { Order = 2 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0},
		{1, 0},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{-20, 0},
		{41./40., -1./40.},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta2NonTVDTableau> RungeKutta2NonTVD; 

struct RungeKutta3TVDTableau {
	enum { Order = 3 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0, 0},
		{3./4., 1./4., 0},
		{1./3., 2./3., 0},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{1, 0, 0},
		{0, 1./4., 0},
		{0, 0, 2./3.},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta3TVDTableau> RungeKutta3TVD; 

struct RungeKutta4TVDTableau {
	enum { Order = 4 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0, 0, 0},
		{649./1600., 951./1600., 0, 0},
		{53989./2500000., 4806213./20000000., 23619./32000., 0},
		{1./5., 6127./30000., 7873./30000., 1./3.},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{.5, 0, 0, 0},
		{-10890423./25193600., 5000./7873, 0, 0},
		{-102261./5000000., -5121./20000., 7873./10000., 0},
		{1./10., 1./6., 0, 1./6.},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta4TVDTableau> RungeKutta4TVD; 

//this one is from http://lsec.cc.ac.cn/lcfd/DEWENO/paper/WENO_1996.pdf

struct RungeKutta4NonTVDTableau {
	enum { Order = 4 };
	static real alphas(int i, int j) { static real m[Order][Order] = {
		{1, 0, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 0, 0},
		{-1./3., 1./3., 2./3., 1./3.},
	}; return m[i][j]; }
	static real betas(int i, int j) { static real m[Order][Order] = {
		{.5, 0, 0, 0},
		{0, .5, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1./6.},
	}; return m[i][j]; }
};
typedef struct RungeKutta<RungeKutta4NonTVDTableau> RungeKutta4NonTVD; 

}
}

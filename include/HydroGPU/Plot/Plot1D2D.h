#pragma once

#include "HydroGPU/Plot/Plot.h"
#include "Tensor/Vector.h"

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot1D2D : public Plot {
	typedef Plot Super;
	
	Plot1D2D(HydroGPU::Solver::Solver& solver);
	
	virtual void resize();
	virtual void mousePan(int dx, int dy);
	virtual void mouseZoom(int dz);
	virtual void screenshot(const std::string& filename);
	
	Tensor::Vector<float,2> viewPos;
	float viewZoom;
};

}
}


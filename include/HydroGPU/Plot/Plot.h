#pragma once

#include <OpenGL/gl.h>
#include <OpenCL/cl.hpp>
#include <memory>
#include <string>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot {
public:	//protected:
	std::shared_ptr<HydroGPU::Solver::Solver> solver;
	
	GLuint tex;
	cl::ImageGL texCLMem;		//data is written to this buffer before rendering
	cl::Kernel convertToTexKernel;

public:
	Plot(std::shared_ptr<HydroGPU::Solver::Solver> solver_);
	virtual ~Plot();

	virtual void init();
	virtual void display();
	virtual void screenshot(const std::string& filename) = 0;

protected:
	virtual void convertVariableToTex(int displayVariable);
};

}
}


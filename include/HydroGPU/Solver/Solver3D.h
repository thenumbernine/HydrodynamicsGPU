#pragma once

#include "HydroGPU/Solver/Solver.h"
#include "Profiler/Stat.h"
#include "Shader/Program.h"
#include "Tensor/Quat.h"
#include <OpenCL/cl.hpp>
#include <vector>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
This class is the remnants of when I had 3 separate CL files for each dimension of the Euler equations.
Each dimension's CL files were instanciated by Solver1D, 2D, and 3D c++ classes, respectively.
Each class also contained the dimension-specific rendering code.
I then merged all the C++ up to the 3D case, got rid of all the <3D CL codes, and haven't moved this file ever since.
Now that the CL codes are merged it might be a good idea to separate the display code into separate 1D, 2D, and 3D classes, 
and move the non-display or non-dimension-specific code back to the Solver class.
*/
struct Solver3D : public Solver {
	typedef Solver Super;
	
	cl::Kernel convertToTexKernel;

	struct EventProfileEntry {
		EventProfileEntry(std::string name_) : name(name_) {}
		std::string name;
		cl::Event clEvent;
		Profiler::Stat stat;
	};
	
	std::vector<EventProfileEntry*> entries;

	cl::ImageGL fluidTexMem;		//data is written to this buffer before rendering
	GLuint fluidTex;

	std::shared_ptr<Shader::Program> displayShader;
	
	GLuint velocityFieldGLBuffer;
	cl::BufferGL velocityFieldVertexBuffer;
	cl::Kernel createVelocityFieldKernel;
	int velocityFieldVertexCount;

	//2D
	Tensor::Vector<float,2> viewPos;
	float viewZoom;
	//3D
	Tensor::Quat<float> viewAngle;
	float viewDist;

	Solver3D(HydroGPUApp &app);
	virtual ~Solver3D();

	virtual void init();

	virtual void display();
	virtual void resize();

	//input
	virtual void mouseMove(int x, int y, int dx, int dy);
	virtual void mousePan(int dx, int dy);
	virtual void mouseZoom(int dz);

	virtual void addDrop();
	virtual void screenshot();
	virtual void save();
};

}
}


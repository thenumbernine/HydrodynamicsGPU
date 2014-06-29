#pragma once

#include "HydroGPU/Solver.h"
#include "Profiler/Stat.h"
#include "Shader/Program.h"
#include "Tensor/Quat.h"
#include <OpenCL/cl.hpp>
#include <vector>
#include <string>

struct HydroGPUApp;

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

	std::shared_ptr<Shader::Program> shader;
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

protected:
	virtual void boundary();
};


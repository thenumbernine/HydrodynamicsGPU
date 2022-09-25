#pragma once

#include "GLCxx/Program.h"
#include "GLCxx/gl.h"
#include "Tensor/Tensor.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Graph {
	HydroGPU::HydroGPUApp * app = {};
	GLCxx::Program graphShader;
	
	Graph(HydroGPU::HydroGPUApp * app_);
	
	/*
	how to display variables ...
	*/
	struct Variable {
		enum PolyMode {
			Point,
			Line,
			Fill,
		};
		
		std::string name;
		bool enabled = false;
		bool log = false;
		int polyMode = PolyMode::Fill;	//0=point, 1=line, 2=fill
		float alpha = 1.f;
		float scale = 1.f;
		int step = 1;
		
		Variable(std::string const & name_)
		: name(name_)
		{}
	};
	std::vector<Variable> variables;

	void display();
};

}
}

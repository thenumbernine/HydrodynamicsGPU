#pragma once

#include "Shader/Program.h"
#include "Tensor/Tensor.h"
#include "Common/gl.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Graph {
	HydroGPU::HydroGPUApp* app;	
	std::shared_ptr<Shader::Program> graphShader;
	
	Graph(HydroGPU::HydroGPUApp* app_);
	
	/*
	how to display variables ...
	*/
	struct Variable {
		typedef enum {
			Point,
			Line,
			Fill,
		} PolyMode;
		
		std::string name;
		bool enabled;
		bool log;
		int polyMode;	//0=point, 1=line, 2=fill
		float alpha;
		float scale;
		int step;
		
		Variable(const std::string& name_)
		: name(name_)
		, enabled(false)
		, log(false)
		, polyMode(PolyMode::Fill)
		, alpha(1.f)
		, scale(1.f)
		, step(1)
		{}
	};
	std::vector<Variable> variables;

	virtual void display();
};

}
}

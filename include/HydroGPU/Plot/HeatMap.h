#pragma once

#include "Shader/Program.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct HeatMap {
protected:
	HydroGPUApp* app;
	std::shared_ptr<Shader::Program> heatShader;

public:
	int variable;
	float scale;
	bool useLog;

	HeatMap(HydroGPU::HydroGPUApp* app_);

	virtual void display();
};

}
}

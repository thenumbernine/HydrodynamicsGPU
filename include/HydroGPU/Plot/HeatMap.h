#pragma once

#include "GLCxx/Program.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct HeatMap {
protected:
	HydroGPUApp* app;
	std::shared_ptr<GLCxx::Program> heatShader;

public:
	int variable;
	float scale;
	bool useLog;
	float alpha;

	HeatMap(HydroGPU::HydroGPUApp* app_);

	virtual void display();
};

}
}

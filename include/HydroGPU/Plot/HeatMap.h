#pragma once

#include "GLCxx/Program.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct HeatMap {
protected:
	HydroGPUApp * app = {};
	GLCxx::Program heatShader;

public:
	int variable = {};
	float scale = 1.f;
	bool useLog = false;
	float alpha = 1.f;

	HeatMap(HydroGPU::HydroGPUApp * app_);

	void display();
};

}
}

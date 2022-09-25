#pragma once

#include "HydroGPU/Plot/Iso3D.h"
#include "GLCxx/Program.h"
#include "CLCommon/cl.hpp"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Iso3D {
protected:
	HydroGPUApp * app = {};
	GLCxx::Program shader;

public:
	int variable = 0;
	float scale = 1.f;
	bool useLog = false;
	float alpha = .5f;

	Iso3D(HydroGPU::HydroGPUApp* app_);

	void display();
};

}
}

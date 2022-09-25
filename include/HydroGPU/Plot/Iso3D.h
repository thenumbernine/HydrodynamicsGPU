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
	HydroGPUApp* app;
	std::shared_ptr<GLCxx::Program> shader;

public:
	int variable;
	float scale;
	bool useLog;
	float alpha;

	Iso3D(HydroGPU::HydroGPUApp* app_);

	void display();
};

}
}

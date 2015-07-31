#pragma once

#include "HydroGPU/Plot/Plot1D2D.h"
#include "Shader/Program.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Plot2D : public Plot1D2D {
	typedef Plot1D2D Super;
	
	std::shared_ptr<Shader::Program> heatShader;

	Plot2D(HydroGPU::HydroGPUApp* app_);

	virtual void display();
};

}
}


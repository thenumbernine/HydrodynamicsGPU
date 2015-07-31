#pragma once

#include "HydroGPU/Plot/Plot1D2D.h"
#include "Shader/Program.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Plot1D : public Plot1D2D {
	typedef Plot1D2D Super;

	Plot1D(HydroGPU::HydroGPUApp* app_);

	virtual void display();

	std::shared_ptr<Shader::Program> displayShader;
};

}
}


#pragma once

#include "HydroGPU/Plot/Plot.h"
#include "Shader/Program.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Plot3D : public Plot {
	typedef Plot Super;
	
	std::shared_ptr<Shader::Program> displayShader;

	Plot3D(HydroGPU::HydroGPUApp* app_);
	
	virtual void display();
};

}
}


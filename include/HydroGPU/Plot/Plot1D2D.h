#pragma once

#include "HydroGPU/Plot/Plot.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Plot1D2D : public Plot {
	typedef Plot Super;
	
	Plot1D2D(HydroGPU::HydroGPUApp* app_);
	
	virtual void screenshot(const std::string& filename);
};

}
}


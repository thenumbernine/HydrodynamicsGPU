#pragma once

#include "HydroGPU/Plot/Iso3D.h"
#include "Shader/Program.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Iso3D {
protected:
	HydroGPUApp* app;
	std::shared_ptr<Shader::Program> shader;
	GLuint tex;

public:
	int variable;
	float scale;
	bool useLog;

	Iso3D(HydroGPU::HydroGPUApp* app_);
	virtual ~Iso3D();

	virtual void display();
};

}
}

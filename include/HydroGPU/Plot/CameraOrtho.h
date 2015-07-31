#pragma once

#include "HydroGPU/Plot/Camera.h"
#include "Tensor/Vector.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct CameraOrtho : public Camera {
	typedef Camera Super;

public:	//protected:	
	Tensor::Vector<float,2> pos;
	float zoom;

public:
	CameraOrtho(HydroGPU::HydroGPUApp* app_);

	virtual void setupProjection();
	virtual void setupModelview();
	virtual void mousePan(int dx, int dy);
	virtual void mouseZoom(int dz);
};

}
}

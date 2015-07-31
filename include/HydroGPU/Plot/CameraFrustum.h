#pragma once

#include "HydroGPU/Plot/Camera.h"
#include "Tensor/Vector.h"
#include "Tensor/Quat.h"
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct CameraFrustum : public Camera {
	typedef Camera Super;

protected:	
	Tensor::Vector<float,3> pos;
	Tensor::Quat<float> angle;
	float dist;

public:
	CameraFrustum(HydroGPU::HydroGPUApp* app_);

	virtual void setupProjection();
	virtual void setupModelview();
	virtual void mousePan(int dx, int dy);
	virtual void mouseZoom(int dz);
};

}
}

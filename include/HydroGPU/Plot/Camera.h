/*
superclass of ortho and frustum views
*/

#pragma once

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Camera {
	HydroGPU::HydroGPUApp* app;

	Camera(HydroGPU::HydroGPUApp* app_) 
	: app(app_)
	{}
	
	virtual void setupProjection() {}
	virtual void setupModelview() {}
	virtual void mousePan(int dx, int dy) {}
	virtual void mouseZoom(int dz) {}
};

}
}

#include "HydroGPU/Plot/CameraOrtho.h"
#include "HydroGPU/HydroGPUApp.h"
#include <OpenGL/gl.h>

namespace HydroGPU {
namespace Plot {

CameraOrtho::CameraOrtho(HydroGPU::HydroGPUApp* app_)
: Super(app_)
, zoom(1.f)
{
	if (!app->lua["camera"]["pos"].isNil()) {
		app->lua["camera"]["pos"][1] >> pos(0);
		app->lua["camera"]["pos"][2] >> pos(1);
	}
	app->lua["camera"]["zoom"] >> zoom;
}

void CameraOrtho::setupProjection() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(
		pos(0) - app->aspectRatio * .5 / zoom, 
		pos(0) + app->aspectRatio * .5 / zoom,
		pos(1) - .5 / zoom,
		pos(1) + .5 / zoom, -1., 1.);
}

void CameraOrtho::setupModelview() {
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void CameraOrtho::mousePan(int dx, int dy) {
	pos += Tensor::Vector<float,2>(
		-(float)dx * app->aspectRatio / (float)app->screenSize(0),
		(float)dy / (float)app->screenSize(1)
	) / zoom;
}

void CameraOrtho::mouseZoom(int dz) {
	float scale = exp((float)dz * -.03f);
	zoom *= scale;
}


}
}

#include "HydroGPU/Plot/CameraOrtho.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/gl.h"

namespace HydroGPU {
namespace Plot {

CameraOrtho::CameraOrtho(HydroGPU::HydroGPUApp* app_)
: Super(app_)
, zoom(1,1)
{
	if (!app->lua["camera"]["pos"].isNil()) {
		app->lua["camera"]["pos"][1] >> pos(0);
		app->lua["camera"]["pos"][2] >> pos(1);
	}
	app->lua["camera"]["zoom"] >> zoom(0);
	zoom(1) = zoom(0);
}

void CameraOrtho::setupProjection() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(
		pos(0) - app->aspectRatio * .5 / zoom(0), 
		pos(0) + app->aspectRatio * .5 / zoom(0),
		pos(1) - .5 / zoom(1),
		pos(1) + .5 / zoom(1), -1., 1.);
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

void CameraOrtho::mouseZoom(int dx, int dy) {
	zoom(0) *= exp(-dx * -.03);
	zoom(1) *= exp(dy * -.03);
}

}
}

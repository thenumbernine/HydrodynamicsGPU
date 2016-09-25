#include "HydroGPU/Plot/CameraFrustum.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/gl.h"

namespace HydroGPU {
namespace Plot {

CameraFrustum::CameraFrustum(HydroGPU::HydroGPUApp* app_)
: Super(app_)
, dist(1.f)
{
	if (!app->lua["camera"]["pos"].isNil()) {
		app->lua["camera"]["pos"][1] >> pos(0);
		app->lua["camera"]["pos"][2] >> pos(1);
		app->lua["camera"]["pos"][3] >> pos(2);
	}
	app->lua["camera"]["dist"] >> dist;
}

void CameraFrustum::setupProjection() {
	float zFar = 10.;
	float zNear = .001;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-app->aspectRatio * zNear, app->aspectRatio * zNear, -zNear, zNear, zNear, zFar);
}

void CameraFrustum::setupModelview() {
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0,0,-dist);
	Tensor::Quat<float> angleAxis = angle.toAngleAxis();
	glRotatef(angleAxis(3) * 180. / M_PI, angleAxis(0), angleAxis(1), angleAxis(2));
	glTranslatef(-pos(0), -pos(1), -pos(2));
}

void CameraFrustum::mousePan(int dx, int dy) {
	float magn = sqrt(dx * dx + dy * dy);
	float fdx = (float)dx / magn;
	float fdy = (float)dy / magn;
	Tensor::Quat<float> rotation = Tensor::Quat<float>(fdy, fdx, 0, magn * M_PI / 180.).fromAngleAxis();
	angle = rotation * angle;
	angle /= Tensor::Quat<float>::length(angle);
}

void CameraFrustum::mouseZoom(int dx, int dy) {
	dist *= (float)exp((float)dy * -.03f);
}

}
}

#include "HydroGPU/Plot/Iso3D.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/Image.h"
#include "GLCxx/gl.h"
#include "Common/File.h"

static float vertexes[] = {
	0,0,0,
	1,0,0,
	0,1,0,
	1,1,0,
	0,0,1,
	1,0,1,
	0,1,1,
	1,1,1,
};

static int quads[] = {
	0,1,3,2,
	4,6,7,5,
	1,5,7,3,
	0,2,6,4,
	0,4,5,1,
	2,3,7,6,
};

namespace HydroGPU {
namespace Plot {

Iso3D::Iso3D(HydroGPU::HydroGPUApp* app_)
: app(app_)
{
	std::string shaderCode = Common::File::read("Isosurface3D.shader");
	shader = GLCxx::Program(
			std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode},
			std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode}
		)
		.setUniform<int>("tex", 0)
		.setUniform<int>("gradient", 1)
		.setUniform<int>("maxiter", std::max(app->size.s[0], std::max(app->size.s[1], app->size.s[2])))
		.setUniform<float>("oneOverDx", app->xmax.s[0] - app->xmin.s[0], app->xmax.s[1] - app->xmin.s[1], app->xmax.s[2] - app->xmin.s[2])
		.done();
}
	
void Iso3D::display() {
	app->plot->convertVariableToTex(variable);
	auto tex = app->plot->getTex();

	glColor3f(1,1,1);
	for (int pass = 0; pass < 2; ++pass) {
		if (pass == 0) {
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		} else {
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);
			glEnable(GL_DEPTH_TEST);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_BLEND);
			shader
				.use()
				.setUniform<float>("scale", scale)
				.setUniform<bool>("useLog", useLog)
				.setUniform<float>("alpha", alpha);
			tex.bind(0);
			app->gradientTex.bind(1);
		}
		glBegin(GL_QUADS);
		for (int i = 0; i < 24; ++i) {
			float x = vertexes[quads[i] * 3 + 0];
			float y = vertexes[quads[i] * 3 + 1];
			float z = vertexes[quads[i] * 3 + 2];
			glTexCoord3f(x, y, z);
			x = x * (app->xmax.s[0] - app->xmin.s[0]) + app->xmin.s[0];
			y = y * (app->xmax.s[1] - app->xmin.s[1]) + app->xmin.s[1];
			z = z * (app->xmax.s[2] - app->xmin.s[2]) + app->xmin.s[2];
			glVertex3f(x, y, z);
		}
		glEnd();
		if (pass == 0) {
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		} else {
			app->gradientTex.unbind(1);
			tex.unbind(0);
			shader.done();
			glDisable(GL_BLEND);
			glDisable(GL_DEPTH_TEST);
			glCullFace(GL_BACK);
			glDisable(GL_CULL_FACE);
		}
	}
}

}
}

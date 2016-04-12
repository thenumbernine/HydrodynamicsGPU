#include "HydroGPU/Plot/Iso3D.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Common/File.h"
#include <OpenGL/gl.h>

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
, variable(0)
, scale(1.f)
, useLog(false)
{
	std::string shaderCode = Common::File::read("Isosurface3D.shader");
	std::vector<Shader::Shader> shaders = {
		Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
		Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
	};
	shader = std::make_shared<Shader::Program>(shaders);
	shader->link()
		.setUniform<int>("tex", 0)
		.setUniform<int>("gradient", 1)
		.setUniform<int>("maxiter", std::max(app->size.s[0], std::max(app->size.s[1], app->size.s[2])))
		.setUniform<float>("oneOverDx", app->xmax.s[0] - app->xmin.s[0], app->xmax.s[1] - app->xmin.s[1], app->xmax.s[2] - app->xmin.s[2])
		.done();
}
	
void Iso3D::display() {
	app->plot->convertVariableToTex(variable);

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
			shader->use()
				.setUniform<float>("scale", scale)
				.setUniform<bool>("useLog", useLog);
			assert(app->plot->getTarget() == GL_TEXTURE_3D);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, app->plot->getTex());
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_1D, app->gradientTex);
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
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_1D, 0);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, 0);
			shader->done();
			glDisable(GL_BLEND);
			glDisable(GL_DEPTH_TEST);
			glCullFace(GL_BACK);
			glDisable(GL_CULL_FACE);
		}
	}
}

}
}

#include "HydroGPU/Plot/Plot2D.h"
#include "HydroGPU/Plot/Camera.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"
#include <OpenGL/gl.h>

namespace HydroGPU {
namespace Plot {
	
Plot2D::Plot2D(std::shared_ptr<HydroGPU::Solver::Solver> solver)
: Super(solver)
{
	std::string shaderCode = Common::File::read("HeatMap.shader");
	std::vector<Shader::Shader> shaders = {
		Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
		Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
	};
	heatShader = std::make_shared<Shader::Program>(shaders);
	heatShader->link()
		.setUniform<int>("tex", 0)
		.setUniform<int>("gradient", 1)
		.done();
}

void Plot2D::display() {
	Super::display();

	solver->app->camera->setupModelview();

	heatShader->use();
	heatShader->setUniform<float>("scale", solver->app->heatMapColorScale);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, solver->app->gradientTex);
	glBegin(GL_QUADS);
	const float xofs = 0.f;
	const float yofs = 0.f;
	glTexCoord2f(0+xofs,0+yofs); glVertex2f(solver->app->xmin.s[0], solver->app->xmin.s[1]);
	glTexCoord2f(1+xofs,0+yofs); glVertex2f(solver->app->xmax.s[0], solver->app->xmin.s[1]);
	glTexCoord2f(1+xofs,1+yofs); glVertex2f(solver->app->xmax.s[0], solver->app->xmax.s[1]);
	glTexCoord2f(0+xofs,1+yofs); glVertex2f(solver->app->xmin.s[0], solver->app->xmax.s[1]);
	glEnd();
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	heatShader->done();
}

}
}


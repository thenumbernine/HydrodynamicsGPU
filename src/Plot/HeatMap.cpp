#include "HydroGPU/Plot/HeatMap.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"
#include <OpenGL/gl.h>
#include <cassert>

namespace HydroGPU {
namespace Plot {
	
HeatMap::HeatMap(HydroGPU::HydroGPUApp* app_)
: app(app_)
, variable(0)
, scale(1.f)
, useLog(false)
, alpha(1.f)
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

void HeatMap::display() {
	app->plot->convertVariableToTex(variable);

	heatShader->use();
	heatShader->setUniform<float>("scale", scale)
				.setUniform<bool>("useLog", useLog)
				.setUniform<float>("alpha", alpha);

	assert(app->plot->getTarget() == GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, app->plot->getTex());
//TODO heat map flag for texture mag filtering
//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, app->gradientTex);
	glBegin(GL_QUADS);
	const float xofs = 0.f;
	const float yofs = 0.f;
	glTexCoord2f(0+xofs,0+yofs); glVertex2f(app->xmin.s[0], app->xmin.s[1]);
	glTexCoord2f(1+xofs,0+yofs); glVertex2f(app->xmax.s[0], app->xmin.s[1]);
	glTexCoord2f(1+xofs,1+yofs); glVertex2f(app->xmax.s[0], app->xmax.s[1]);
	glTexCoord2f(0+xofs,1+yofs); glVertex2f(app->xmin.s[0], app->xmax.s[1]);
	glEnd();
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	heatShader->done();
}

}
}

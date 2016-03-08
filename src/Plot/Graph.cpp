#include "HydroGPU/Plot/Graph.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Plot/Camera.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Macros.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Plot {

Graph::Graph(HydroGPU::HydroGPUApp* app_)
: app(app_)
, scale(1.f)
{
	std::string shaderCode = Common::File::read("Graph.shader");
	std::vector<Shader::Shader> shaders = {
		Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
		Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
	};
	graphShader = std::make_shared<Shader::Program>(shaders);
	graphShader->link();
	graphShader->setUniform<int>("tex", 0);
	graphShader->setUniform<float>("xmin", app->xmin.s[0], app->xmin.s[1]);
	graphShader->setUniform<float>("xmax", app->xmax.s[0], app->xmax.s[1]);
	graphShader->done();
}

void Graph::display() {
if (app->dim == 1) return;	//for now plot1d does the same thing ...

	app->camera->setupModelview();

	static float colors[][3] = {
		{1,0,0},
		{0,1,0},
		{0,.5,1},
		{1,.5,0}
	};

	// TODO only select variables? 
	for (int var : variables) {
		app->plot->convertVariableToTex(var);
		
		graphShader->use();
		graphShader->setUniform<float>("scale", scale)
					.setUniform<int>("axis", app->dim);
		glBindTexture(GL_TEXTURE_2D, app->plot->tex);
		glColor3fv(colors[var % numberof(colors)]);
		
		switch (app->dim) {
		case 1:
			glBegin(GL_LINE_STRIP);
			for (int i = 2; i < app->size.s[0]-2; ++i) {
				real x = ((real)(i) + .5f) / (real)app->size.s[0];
				glVertex2f(x, 0.f);
			}
			glEnd();
			break;
		case 2:
			{
				Tensor::Vector<float,2> pt;
				for (int k = 0; k < app->dim; ++k) {
					for (int j = 2; j < app->size.s[!k]-2; j += step(k)) {
						pt(!k) = ((real)(j) + .5f) / (real)app->size.s[!k];
						glBegin(GL_LINE_STRIP);
						for (int i = 2; i < app->size.s[k]-2; ++i) {
							pt(k) = ((real)(i) + .5f) / (real)app->size.s[k];
							glVertex2f(pt(0), pt(1));
						}
						glEnd();
					}
				}
			}
			break;
		case 3:
			break;
		}
		
		graphShader->done();
		glBindTexture(GL_TEXTURE_2D, 0);	
	}
}

}
}

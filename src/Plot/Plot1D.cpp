#include "HydroGPU/Plot/Plot1D.h"
#include "HydroGPU/Plot/CameraOrtho.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"
#include "Common/Macros.h"
#include <OpenGL/gl.h>

namespace HydroGPU {
namespace Plot {
	
Plot1D::Plot1D(HydroGPU::HydroGPUApp* app_) 
: Super(app_)
{
	std::string shaderCode = Common::File::read("Display1D.shader");
	std::vector<Shader::Shader> shaders = {
		Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
		Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
	};
	displayShader = std::make_shared<Shader::Program>(shaders);
	displayShader->link();
	displayShader->setUniform<int>("tex", 0);
	displayShader->setUniform<float>("xmin", app->xmin.s[0]);
	displayShader->setUniform<float>("xmax", app->xmax.s[0]);
	displayShader->done();	
}

void Plot1D::display() {

	static float colors[][3] = {
		{1,0,0},
		{0,1,0},
		{0,.5,1},
		{1,.5,0}
	};

	//determine grid for width
	//render lines

	for (int variableIndex = 0; variableIndex < app->solver->equation->displayVariables.size(); ++variableIndex) {

		convertVariableToTex(variableIndex);

		displayShader->use();
		glBindTexture(GL_TEXTURE_2D, tex);
		glColor3fv(colors[variableIndex % numberof(colors)]);
		displayShader->setUniform<float>("scale", app->heatMapColorScale);
		glBegin(GL_LINE_STRIP);
		for (int i = 2; i < app->size.s[0]-2; ++i) {
			real x = ((real)(i) + .5f) / (real)app->size.s[0];
			//incoming vertexes must be within [0,1] for texture loop
			//then they are rescaled to the graph bounds
			glVertex2f(x, 0.f);
		}
		glEnd();
		displayShader->done();
		glBindTexture(GL_TEXTURE_2D, 0);	
	}

	{
		Tensor::Vector<double,2> viewxmax(app->aspectRatio * .5, .5);
		Tensor::Vector<double,2> viewxmin = -viewxmax;
		std::shared_ptr<CameraOrtho> cameraOrtho = std::dynamic_pointer_cast<CameraOrtho>(app->camera);
		if (cameraOrtho) {
			viewxmin /= cameraOrtho->zoom;
			viewxmax /= cameraOrtho->zoom;
			viewxmin += cameraOrtho->pos;
			viewxmax += cameraOrtho->pos;
		}
		double spacing = std::max( viewxmax(0) - viewxmin(0), viewxmax(1) - viewxmin(1) );
		spacing = pow(10.,floor(log10(spacing))) * .1;
		for (int i = 0; i < 2; ++i) {
			viewxmin(i) = spacing * floor(viewxmin(i) / spacing);
			viewxmax(i) = spacing * ceil(viewxmax(i) / spacing);
		}
	
		glBegin(GL_LINES);
		glColor3f(.5, .5, .5);
		glVertex2f(0, viewxmin(1));
		glVertex2f(0, viewxmax(1));
		glVertex2f(viewxmin(0), 0);
		glVertex2f(viewxmax(0), 0);
		glColor3f(.25, .25, .25);
		for (double x = viewxmin(0); x <= viewxmax(0); x += spacing) {
			glVertex2f(x, viewxmin(1));
			glVertex2f(x, viewxmax(1));
		}
		for (double y = viewxmin(1); y < viewxmax(1); y += spacing) {
			glVertex2f(viewxmin(0), y);
			glVertex2f(viewxmax(0), y);
		}
		glEnd();
	}
}

}
}


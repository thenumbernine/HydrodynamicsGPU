#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Plot/Graph.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Plot/CameraOrtho.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver/Solver.h"
#include "Common/Macros.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Plot {

Graph::Graph(HydroGPU::HydroGPUApp* app_)
: app(app_)
{
	std::string shaderCode = Common::File::read("Graph.shader");
	std::vector<Shader::Shader> shaders = {
		Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
		Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
	};
	graphShader = std::make_shared<Shader::Program>(shaders);
	graphShader->link();
	graphShader->setUniform<int>("tex", 0);
	graphShader->done();

	variables.clear();
	for (const std::string& name : app->solver->equation->displayVariables) {
		variables.push_back(Variable(name));
	}
}

void Graph::display() {
	
	//if we're in ortho mode and we're 2D then cut out
	if (app->dim == 2 && std::dynamic_pointer_cast<CameraOrtho>(app->camera)) return;
	
	static float colors[][3] = {
		{1,0,0},
		{1,1,0},
		{0,1,0},
		{0,1,1},
		{0,0,1},
		{1,0,1},
	};
	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	for (int drawAlpha = 0; drawAlpha <= 1; ++drawAlpha) {
		if (drawAlpha) {
			glEnable(GL_BLEND);
			glDepthMask(GL_FALSE);
		}
	
		for (int i = 0; i < variables.size(); ++i) {
			const Variable& var = variables[i];
			if (!var.enabled) continue;
			if (var.alpha < 1 != drawAlpha) continue;
		
			int step = var.step;
			if (step < 1) step = 1;
		
			app->plot->convertVariableToTex(i);
			
			graphShader->use();
			graphShader->setUniform<float>("scale", var.scale)
						.setUniform<int>("axis", app->dim)
						.setUniform<float>("ambient", app->dim == 1 ? 1 : .4)
						.setUniform<bool>("useLog", var.log)
						.setUniform<float>("size", app->size.s[0], app->size.s[1]);
			graphShader->setUniform<float>("xmin", app->xmin.s[0], app->xmin.s[1]);
			graphShader->setUniform<float>("xmax", app->xmax.s[0], app->xmax.s[1]);
			glBindTexture(GL_TEXTURE_2D, app->plot->getTex());
			const float* c = colors[i % numberof(colors)];
			glColor4f(c[0], c[1], c[2], var.alpha);
			
			switch (app->dim) {
			case 1:
				glBegin(GL_LINE_STRIP);
				for (int i = 2; i < app->size.s[0]-2; i += step) {
					real x = ((real)(i) + .5f) / (real)app->size.s[0];
					glVertex2f(x, 0.f);
				}
				glEnd();
				break;
			case 2:
				{
					switch (var.polyMode) {
					case Variable::PolyMode::Point: glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); break;
					case Variable::PolyMode::Line: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); break;
					case Variable::PolyMode::Fill: default: glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); break;
					}

					Tensor::Vector<float,2> pt;
					for (int j = 2; j < app->size.s[1]-2-step; j += step) {
						glBegin(GL_TRIANGLE_STRIP);
						for (int i = 2; i < app->size.s[0]-2; i += step) {
							pt(0) = ((real)(i) + .5f) / (real)app->size.s[0];
							for (int jofs = step; jofs >= 0; jofs -= step) {
								pt(1) = ((real)(j + jofs) + .5f) / (real)app->size.s[1];
								glVertex2f(pt(0), pt(1));
							}
						}
						glEnd();
					}
				
					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				}
				break;
			case 3:
				break;
			}
			
			graphShader->done();
			glBindTexture(GL_TEXTURE_2D, 0);	
		}
		if (drawAlpha) {
			glDisable(GL_BLEND);
			glDepthMask(GL_TRUE);
		}
	}

	//for 1-D graphs
	//draw background grid and axii
	if (app->dim == 1) {
		Tensor::Vector<double,2> viewxmax(app->aspectRatio * .5, .5);
		Tensor::Vector<double,2> viewxmin = -viewxmax;
		std::shared_ptr<CameraOrtho> cameraOrtho = std::dynamic_pointer_cast<CameraOrtho>(app->camera);
		if (cameraOrtho) {
			viewxmin /= cameraOrtho->zoom;
			viewxmax /= cameraOrtho->zoom;
			viewxmin += cameraOrtho->pos;
			viewxmax += cameraOrtho->pos;
		
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
}

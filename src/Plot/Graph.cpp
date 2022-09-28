#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Plot/Graph.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver/Solver.h"
#include "GLApp/GLApp.h"
#include "GLApp/ViewOrtho.h"
#include "GLCxx/gl.h"
#include "Common/Macros.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Plot {

Graph::Graph(HydroGPU::HydroGPUApp* app_)
: app(app_)
{
	std::string shaderCode = Common::File::read("Graph.shader");
	graphShader = GLCxx::Program(
		std::vector<std::string>{
			"#define VERTEX_SHADER\n",
			shaderCode,
		},
		std::vector<std::string>{
			"#define FRAGMENT_SHADER\n",
			shaderCode,
		}
	)
		.setUniform<int>("tex", 0)
		.done();

	variables.clear();
	for (const std::string& name : app->solver->equation->displayVariables) {
		variables.push_back(Variable(name));
	}
}

void Graph::display() {
	
	//if we're in ortho mode and we're 2D then cut out
	if (app->dim == 2 && std::dynamic_pointer_cast<GLApp::ViewOrtho>(app->view)) return;
	
	static float colors[][3] = {
		{1,0,0},
		{1,1,0},
		{0,1,0},
		{0,1,1},
		{.25,.5,1},
		{1,0,1},
	};
	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	for (int drawAlpha = 0; drawAlpha <= 1; ++drawAlpha) {
		if (drawAlpha) {
			glEnable(GL_BLEND);
			glDepthMask(GL_FALSE);
		}
		
		for (int i = 0; i < (int)variables.size(); ++i) {
			const Variable& var = variables[i];
			if (!var.enabled) continue;
			if ((var.alpha < 1) != drawAlpha) continue;
		
			int step = var.step;
			if (step < 1) step = 1;
		
			app->plot->convertVariableToTex(i);
			
			graphShader
				.use()
				.setUniform<float>("scale", var.scale)
				.setUniform<int>("axis", app->dim)
				.setUniform<float>("ambient", app->dim == 1 ? 1 : .4)
				.setUniform<bool>("useLog", var.log)
				.setUniform<float>("size", app->size.s[0], app->size.s[1])
				.setUniform<float>("xmin", app->xmin.s[0], app->xmin.s[1])
				.setUniform<float>("xmax", app->xmax.s[0], app->xmax.s[1]);
			auto tex = app->plot->getTex();
			tex.bind();
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

					Tensor::float2 pt;
					for (int j = 2; j < app->size.s[1]-2-step; j += step) {
						glBegin(GL_TRIANGLE_STRIP);
						for (int i = 2; i < app->size.s[0]-2; i += step) {
							pt.x = ((real)i + (real).5) / (real)app->size.s[0];
							for (int jofs = step; jofs >= 0; jofs -= step) {
								pt.y = ((real)(j + jofs) + (real).5) / (real)app->size.s[1];
								glVertex2f(pt.x, pt.y);
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
			
			graphShader.done();
			tex.unbind();
		}
		if (drawAlpha) {
			glDisable(GL_BLEND);
			glDepthMask(GL_TRUE);
		}
	}

	//for 1-D graphs
	//draw background grid and axii
	if (app->dim == 1) {
		Tensor::double2 viewxmax(app->getAspectRatio() * .5, .5);
		Tensor::double2 viewxmin = -viewxmax;
		std::shared_ptr<GLApp::ViewOrtho> viewOrtho = std::dynamic_pointer_cast<GLApp::ViewOrtho>(app->view);
		if (viewOrtho) {
			viewxmin.x /= viewOrtho->zoom.x;
			viewxmin.y /= viewOrtho->zoom.y;
			viewxmax.x /= viewOrtho->zoom.x;
			viewxmax.y /= viewOrtho->zoom.y;
			viewxmin += viewOrtho->pos;
			viewxmax += viewOrtho->pos;

			Tensor::double2 spacing;
			spacing.x = viewxmax.x - viewxmin.x;
			spacing.x = pow(10.,floor(log10(spacing.x))) * .1;
			
			spacing.y = viewxmax.y - viewxmin.y;
			spacing.y = pow(10.,floor(log10(spacing.y))) * .1;
			
			for (int i = 0; i < 2; ++i) {
				viewxmin(i) = spacing(i) * floor(viewxmin(i) / spacing(i));
				viewxmax(i) = spacing(i) * ceil(viewxmax(i) / spacing(i));
			}
		
			glBegin(GL_LINES);
			glColor3f(.5, .5, .5);
			glVertex2f(0, viewxmin.y);
			glVertex2f(0, viewxmax.y);
			glVertex2f(viewxmin.x, 0);
			glVertex2f(viewxmax.x, 0);
			glColor3f(.25, .25, .25);
			for (double x = viewxmin.x; x <= viewxmax.x; x += spacing.x) {
				glVertex2f(x, viewxmin.y);
				glVertex2f(x, viewxmax.y);
			}
			for (double y = viewxmin.y; y < viewxmax.y; y += spacing.y) {
				glVertex2f(viewxmin.x, y);
				glVertex2f(viewxmax.x, y);
			}
			glEnd();
		}
	}
}

}
}

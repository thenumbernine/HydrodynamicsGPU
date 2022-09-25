#include "HydroGPU/Plot/HeatMap.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/HydroGPUApp.h"
#include "GLCxx/gl.h"
#include "Common/File.h"
#include <cassert>

// TODO put this in one place ... separate of all else.  call it "rulers3D" or something ... and put Graph's 2D stuff in "rulers2D"
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
	
HeatMap::HeatMap(HydroGPU::HydroGPUApp* app_)
: app(app_)
, variable(0)
, scale(1.f)
, useLog(false)
, alpha(1.f)
{
	std::string shaderCode = Common::File::read(app->dim < 3 ? "HeatMap2D.shader" : "HeatMap3D.shader");
	std::vector<GLCxx::Shader> shaders = {
		GLCxx::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
		GLCxx::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
	};
	heatShader = std::make_shared<GLCxx::Program>(shaders);
	heatShader->setUniform<int>("tex", 0)
		.setUniform<int>("gradient", 1)
		.done();
}

void HeatMap::display() {
	glColor3f(1,1,1);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
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
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	app->plot->convertVariableToTex(variable);

	heatShader->use();
	heatShader->setUniform<float>("scale", scale)
				.setUniform<bool>("useLog", useLog)
				.setUniform<float>("alpha", alpha);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(app->plot->getTarget(), app->plot->getTex());
//TODO heat map flag for texture mag filtering
//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, app->gradientTex);
	
	if (app->dim < 3) {
		assert(app->plot->getTarget() == GL_TEXTURE_2D);
		const float xofs = 0.f;
		const float yofs = 0.f;
		glBegin(GL_QUADS);
		glTexCoord2f(0+xofs,0+yofs); glVertex2f(app->xmin.s[0], app->xmin.s[1]);
		glTexCoord2f(1+xofs,0+yofs); glVertex2f(app->xmax.s[0], app->xmin.s[1]);
		glTexCoord2f(1+xofs,1+yofs); glVertex2f(app->xmax.s[0], app->xmax.s[1]);
		glTexCoord2f(0+xofs,1+yofs); glVertex2f(app->xmin.s[0], app->xmax.s[1]);
		glEnd();
	} else {
		assert(app->plot->getTarget() == GL_TEXTURE_3D);
		//point cloud ... at what spacing?
		//TODO use a buffer or a call list
		glPointSize(5);
		glBegin(GL_POINTS);
		for (int i = 0; i < app->size.s[0]; ++i) {
			float x = ((float)i+.5f)/(float)app->size.s[0];
			float ix = 1.f - x;
			for (int j = 0; j < app->size.s[1]; ++j) {
				float y = ((float)j+.5f)/(float)app->size.s[1];
				float iy = 1.f - y;
				for (int k = 0; k < app->size.s[2]; ++k) {
					float z = ((float)k+.5f)/(float)app->size.s[2];
					float iz = 1.f - z;
					glTexCoord3f(x,y,z);
					glVertex3f(
						ix * app->xmin.s[0] + x * app->xmax.s[0],
						iy * app->xmin.s[1] + y * app->xmax.s[1],
						iz * app->xmin.s[2] + z * app->xmax.s[2]);
				}
			}
		}
		glEnd();	
		glPointSize(1);
	}
	
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(app->plot->getTarget(), 0);
	heatShader->done();
}

}
}

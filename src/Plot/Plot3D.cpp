#include "HydroGPU/Plot/Plot3D.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Image/FITS_IO.h"
#include "Common/File.h"
#include <OpenGL/gl.h>

static float vertexes[] = {
	0, 0, 0,
	1, 0, 0,
	0, 1, 0,
	1, 1, 0,
	0, 0, 1,
	1, 0, 1,
	0, 1, 1,
	1, 1, 1,
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

Plot3D::Plot3D(HydroGPU::Solver::Solver& solver)
: Super(solver)
, viewDist(1.f)
{
	int volume = solver.getVolume();
	
	std::string shaderCode = Common::File::read("Display3D.shader");
	std::vector<Shader::Shader> shaders = {
		Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
		Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
	};
	displayShader = std::make_shared<Shader::Program>(shaders);
	displayShader->link()
		.setUniform<int>("tex", 0)
		.setUniform<int>("maxiter", std::max(solver.app.size.s[0], std::max(solver.app.size.s[1], solver.app.size.s[2])))
		.setUniform<float>("scale", solver.app.xmax.s[0] - solver.app.xmin.s[0], solver.app.xmax.s[1] - solver.app.xmin.s[1], solver.app.xmax.s[2] - solver.app.xmin.s[2])
		.done();

	//get a texture going for visualizing the output
	glGenTextures(1, &fluidTex);
	glBindTexture(GL_TEXTURE_3D, fluidTex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	Tensor::Vector<int,3> glWraps(GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R);
	//specific to Euler
	for (int i = 0; i < solver.app.dim; ++i) {
		switch (solver.app.boundaryMethods(i)) {
		case 0://BOUNDARY_PERIODIC:
			glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_REPEAT);
			break;
		case 1://BOUNDARY_MIRROR:
		case 2://BOUNDARY_FREEFLOW:
			glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_CLAMP_TO_EDGE);
			break;
		}
	}
	glTexImage3D(GL_TEXTURE_3D, 0, 4, solver.app.size.s[0], solver.app.size.s[1], solver.app.size.s[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	solver.totalAlloc += sizeof(char) * 4 * volume;
	std::cout << "allocating texture size " << (sizeof(float) * 4 * volume) << " running total " << solver.totalAlloc << std::endl;
	glBindTexture(GL_TEXTURE_3D, 0);
	int err = glGetError();
	if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;
}

void Plot3D::display() {
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0,0,-viewDist);
	Tensor::Quat<float> angleAxis = viewAngle.toAngleAxis();
	glRotatef(angleAxis(3) * 180. / M_PI, angleAxis(0), angleAxis(1), angleAxis(2));

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
			displayShader->use();
			glBindTexture(GL_TEXTURE_3D, fluidTex);
		}
		glBegin(GL_QUADS);
		for (int i = 0; i < 24; ++i) {
			float x = vertexes[quads[i] * 3 + 0];
			float y = vertexes[quads[i] * 3 + 1];
			float z = vertexes[quads[i] * 3 + 2];
			glTexCoord3f(x, y, z);
			x = x * (solver.app.xmax.s[0] - solver.app.xmin.s[0]) + solver.app.xmin.s[0];
			y = y * (solver.app.xmax.s[1] - solver.app.xmin.s[1]) + solver.app.xmin.s[1];
			z = z * (solver.app.xmax.s[2] - solver.app.xmin.s[2]) + solver.app.xmin.s[2];
			glVertex3f(x, y, z);
		}
		glEnd();
		if (pass == 0) {
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		} else {
			glBindTexture(GL_TEXTURE_3D, 0);
			displayShader->done();
			glDisable(GL_BLEND);
			glDisable(GL_DEPTH_TEST);
			glCullFace(GL_BACK);
			glDisable(GL_CULL_FACE);
		}
	}
}

void Plot3D::resize() {
	const float zNear = .01;
	const float zFar = 10;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-solver.app.aspectRatio * zNear, solver.app.aspectRatio * zNear, -zNear, zNear, zNear, zFar);
}

void Plot3D::mousePan(int dx, int dy) {
	float magn = sqrt(dx * dx + dy * dy);
	float fdx = (float)dx / magn;
	float fdy = (float)dy / magn;
	Tensor::Quat<float> rotation = Tensor::Quat<float>(fdy, fdx, 0, magn * M_PI / 180.).fromAngleAxis();
	viewAngle = rotation * viewAngle;
	viewAngle /= Tensor::Quat<float>::length(viewAngle);
}

void Plot3D::mouseZoom(int dz) {
	viewDist *= (float)exp((float)dz * -.03f);
}

void Plot3D::screenshot(const std::string& filename) {
	std::shared_ptr<Image::Image> image = std::make_shared<Image::Image>(
		Tensor::Vector<int,2>(solver.app.size.s[0], solver.app.size.s[1]),
		nullptr, 3);
	
	size_t volume = solver.getVolume();
	std::vector<char> buffer(volume);
	glBindTexture(GL_TEXTURE_3D, fluidTex);
	glGetTexImage(GL_TEXTURE_3D, 0, GL_RGB, GL_UNSIGNED_BYTE, &buffer[0]);
	glBindTexture(GL_TEXTURE_3D, 0);
	std::vector<char>::iterator iter = buffer.begin();
	size_t sliceSize = solver.app.size.s[0] * solver.app.size.s[1];
	for (int z = 0; z < solver.app.size.s[2]; ++z) {
		std::copy(iter, iter + sliceSize, image->getData());
		iter += sliceSize;
		std::string layerFilename = filename + "-layer" + std::to_string(z) + ".png";
		Image::system->write(layerFilename, image);
	}
}

}
}


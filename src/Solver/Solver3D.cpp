#include "HydroGPU/Solver/Solver3D.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Image/FITS_IO.h"
#include "Common/File.h"
#include "Common/Exception.h"
#include <OpenGL/gl.h>
#include <iostream>

namespace HydroGPU {
namespace Solver {

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

Solver3D::Solver3D(
	HydroGPUApp &app_)
: Super(app_)
, fluidTex(GLuint())
, velocityFieldGLBuffer(0)
, velocityFieldVertexCount(0)
, viewZoom(1.f)
, viewDist(1.f)
{
}

void Solver3D::init() {
	Super::init();
	
	//memory

	int volume = getVolume();

	switch (app.dim) {
	case 1:
		{
			std::string shaderCode = Common::File::read("Display1D.shader");
			std::vector<Shader::Shader> shaders = {
				Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
				Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
			};
			displayShader = std::make_shared<Shader::Program>(shaders);
			displayShader->link();
			displayShader->setUniform<int>("tex", 0);
			displayShader->setUniform<float>("xmin", app.xmin.s[0]);
			displayShader->setUniform<float>("xmax", app.xmax.s[0]);
			displayShader->done();
		}
		//don't break
	case 2:
		{
			//get a texture going for visualizing the output
			glGenTextures(1, &fluidTex);
			glBindTexture(GL_TEXTURE_2D, fluidTex);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			Tensor::Vector<int,3> glWraps(GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R);
			//specific to Euler
			for (int i = 0; i < app.dim; ++i) {
				switch (app.boundaryMethods(i)) {
				case 0://BOUNDARY_PERIODIC:
					glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_REPEAT);
					break;
				case 1://BOUNDARY_MIRROR:
				case 2://BOUNDARY_FREEFLOW:
					glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_CLAMP_TO_EDGE);
					break;
				}
			}
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, app.size.s[0], app.size.s[1], 0, GL_RGBA, GL_FLOAT, nullptr);
			totalAlloc += sizeof(float) * 4 * volume;
			std::cout << "allocating texture size " << (sizeof(float) * 4 * volume) << " running total " << totalAlloc << std::endl;
			glBindTexture(GL_TEXTURE_2D, 0);
			int err = glGetError();
			if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;
		}
		break;
	case 3:
		{
			std::string shaderCode = Common::File::read("Display3D.shader");
			std::vector<Shader::Shader> shaders = {
				Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
				Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
			};
			displayShader = std::make_shared<Shader::Program>(shaders);
			displayShader->link()
				.setUniform<int>("tex", 0)
				.setUniform<int>("maxiter", std::max(app.size.s[0], std::max(app.size.s[1], app.size.s[2])))
				.setUniform<float>("scale", app.xmax.s[0] - app.xmin.s[0], app.xmax.s[1] - app.xmin.s[1], app.xmax.s[2] - app.xmin.s[2])
				.done();
		
			//get a texture going for visualizing the output
			glGenTextures(1, &fluidTex);
			glBindTexture(GL_TEXTURE_3D, fluidTex);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			Tensor::Vector<int,3> glWraps(GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R);
			//specific to Euler
			for (int i = 0; i < app.dim; ++i) {
				switch (app.boundaryMethods(i)) {
				case 0://BOUNDARY_PERIODIC:
					glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_REPEAT);
					break;
				case 1://BOUNDARY_MIRROR:
				case 2://BOUNDARY_FREEFLOW:
					glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_CLAMP_TO_EDGE);
					break;
				}
			}
			glTexImage3D(GL_TEXTURE_3D, 0, 4, app.size.s[0], app.size.s[1], app.size.s[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			totalAlloc += sizeof(char) * 4 * volume;
			std::cout << "allocating texture size " << (sizeof(float) * 4 * volume) << " running total " << totalAlloc << std::endl;
			glBindTexture(GL_TEXTURE_3D, 0);
			int err = glGetError();
			if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;
		}
		break;
	}

	fluidTexMem = cl::ImageGL(app.context, CL_MEM_WRITE_ONLY, GL_TEXTURE_3D, 0, fluidTex);
	
	convertToTexKernel = cl::Kernel(program, "convertToTex");
	app.setArgs(convertToTexKernel, stateBuffer, potentialBuffer, fluidTexMem, app.gradientTexMem);


	//create GL buffer
	glGenBuffers(1, &velocityFieldGLBuffer);
	glBindBuffer(GL_ARRAY_BUFFER_ARB, velocityFieldGLBuffer);
	int velocityFieldVolume = 1;
	for (int i = 0; i < app.dim; ++i) {
		velocityFieldVolume *= app.velocityFieldResolution;
	}
	velocityFieldVertexCount = 3 * 6 * velocityFieldVolume;
	glBufferData(GL_ARRAY_BUFFER_ARB, sizeof(real) * velocityFieldVertexCount, nullptr, GL_DYNAMIC_DRAW_ARB);
	totalAlloc += sizeof(real) * velocityFieldVertexCount;
	glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	//create CL interop
	velocityFieldVertexBuffer = cl::BufferGL(app.context, CL_MEM_READ_WRITE, velocityFieldGLBuffer);
	//create transfer kernel
	createVelocityFieldKernel = cl::Kernel(program, "createVelocityField");
	app.setArgs(createVelocityFieldKernel, velocityFieldVertexBuffer, stateBuffer, potentialBuffer, app.velocityFieldScale);


	initKernels();
}

Solver3D::~Solver3D() {
	glDeleteTextures(1, &fluidTex);
	glDeleteBuffers(1, &velocityFieldGLBuffer);	
	
	std::cout << "OpenCL profiling info:" << std::endl;
	for (EventProfileEntry *entry : entries) {
		std::cout << entry->name << "\t" 
			<< "duration " << entry->stat
			<< std::endl;
	}
}

void Solver3D::display() {
	glFinish();
	
	std::vector<cl::Memory> acquireGLMems = {fluidTexMem};
	commands.enqueueAcquireGLObjects(&acquireGLMems);

	if (app.useGPU) {
		convertToTexKernel.setArg(4, app.displayMethod);
		convertToTexKernel.setArg(5, app.displayScale);
		commands.enqueueNDRangeKernel(convertToTexKernel, offsetNd, globalSize, localSize);
	} else {
		throw Common::Exception() << "no support";
	}

	commands.enqueueReleaseGLObjects(&acquireGLMems);
	commands.finish();

	switch (app.dim) {
	case 1:
		glLoadIdentity();
		glTranslatef(-viewPos(0), -viewPos(1), 0);
		glScalef(viewZoom, viewZoom, viewZoom);
	
		static float colors[][3] = {
			{1,0,0},
			{0,1,0},
			{0,.5,1},
			{1,.5,0}
		};
	
		//determine grid for width
		//render lines

		glBindTexture(GL_TEXTURE_2D, fluidTex);
		displayShader->use();
		for (int channel = 0; channel < 4; ++channel) {
			glColor3fv(colors[channel]);
			displayShader->setUniform<int>("channel", channel);
			glBegin(GL_LINE_STRIP);
			for (int i = 2; i < app.size.s[0]-2; ++i) {
				real x = ((real)(i) + .5f) / (real)app.size.s[0];
				glVertex2f(x, 0.f);
			}
			glEnd();
		}
		displayShader->done();
		glBindTexture(GL_TEXTURE_2D, 0);	

		{
			Tensor::Vector<double,2> viewxmax(app.aspectRatio * .5, .5);
			Tensor::Vector<double,2> viewxmin = -viewxmax;
			viewxmin += viewPos;
			viewxmax += viewPos;
			viewxmin /= viewZoom;
			viewxmax /= viewZoom;
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
			
		break;
	case 2:
		glLoadIdentity();
		glTranslatef(-viewPos(0), -viewPos(1), 0);
		glScalef(viewZoom, viewZoom, viewZoom);
		
		glBindTexture(GL_TEXTURE_2D, fluidTex);
		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
		glTexCoord2f(0,0); glVertex2f(app.xmin.s[0], app.xmin.s[1]);
		glTexCoord2f(1,0); glVertex2f(app.xmax.s[0], app.xmin.s[1]);
		glTexCoord2f(1,1); glVertex2f(app.xmax.s[0], app.xmax.s[1]);
		glTexCoord2f(0,1); glVertex2f(app.xmin.s[0], app.xmax.s[1]);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
	
		break;
	case 3:
		{
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
					x = x * (app.xmax.s[0] - app.xmin.s[0]) + app.xmin.s[0];
					y = y * (app.xmax.s[1] - app.xmin.s[1]) + app.xmin.s[1];
					z = z * (app.xmax.s[2] - app.xmin.s[2]) + app.xmin.s[2];
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
		break;
	}

	if (app.showVelocityField) {
	
		//glFlush();
		cl::NDRange global;
		switch (app.dim) {
		case 1:
			global = cl::NDRange(app.velocityFieldResolution);
			break;
		case 2:
			global = cl::NDRange(app.velocityFieldResolution, app.velocityFieldResolution);
			break;
		case 3:
			global = cl::NDRange(app.velocityFieldResolution, app.velocityFieldResolution, app.velocityFieldResolution);
			break;
		}
		createVelocityFieldKernel.setArg(3, app.velocityFieldScale);
		commands.enqueueNDRangeKernel(createVelocityFieldKernel, offsetNd, global, localSize);
		commands.finish();

		glDisable(GL_DEPTH_TEST);

		glBindBuffer(GL_ARRAY_BUFFER_ARB, velocityFieldGLBuffer);
		glColor3d(1,1,1);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glDrawArrays(GL_LINES, 0, velocityFieldVertexCount);
		glDisableClientState(GL_VERTEX_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
		
		glEnable(GL_DEPTH_TEST);
	}


	{int err = glGetError();
	if (err) std::cout << "GL error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl;}
}

void Solver3D::resize() {
	const float zNear = .01;
	const float zFar = 10;
	switch (app.dim) {
	case 1:
	case 2:
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-app.aspectRatio *.5, app.aspectRatio * .5, -.5, .5, -1., 1.);
		glMatrixMode(GL_MODELVIEW);
		break;
	case 3:
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-app.aspectRatio * zNear, app.aspectRatio * zNear, -zNear, zNear, zNear, zFar);
		break;
	}
}

void Solver3D::addDrop() {
}

void Solver3D::screenshot() {
	for (int i = 0; i < 1000; ++i) {
		std::string filename = std::string("screenshot") + std::to_string(i) + "layer0.png";
		if (!Common::File::exists(filename)) {
			std::shared_ptr<Image::Image> image = std::make_shared<Image::Image>(
				Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]),
				nullptr, 3);
			switch (app.dim) {
			case 1:
				std::cout << "take a picture, it'll last longer" << std::endl;
				break;
			case 2:
				glBindTexture(GL_TEXTURE_2D, fluidTex);
				glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, image->getData());
				glBindTexture(GL_TEXTURE_2D, 0);
				Image::system->write(filename, image);
				break;
			case 3:
				size_t volume = getVolume();
				std::vector<char> buffer(volume);
				glBindTexture(GL_TEXTURE_3D, fluidTex);
				glGetTexImage(GL_TEXTURE_3D, 0, GL_RGB, GL_UNSIGNED_BYTE, &buffer[0]);
				glBindTexture(GL_TEXTURE_3D, 0);
				std::vector<char>::iterator iter = buffer.begin();
				size_t sliceSize = app.size.s[0] * app.size.s[1];
				for (int z = 0; z < app.size.s[2]; ++z) {
					std::copy(iter, iter + sliceSize, image->getData());
					iter += sliceSize;
					filename = std::string("screenshot") + std::to_string(i) + "layer" + std::to_string(z) + ".png";
					Image::system->write(filename, image);
				}
				break;
			}
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

//TODO this should be handled per-equation
void Solver3D::save() {
	std::vector<std::string> channelNames = equation->states;
	channelNames.push_back("potential");

	for (int i = 0; i < 1000; ++i) {
		std::string filename = channelNames[0] + std::to_string(i) + ".fits";
		if (!Common::File::exists(filename)) {
			
			//hmm, rather than a plane per variable, now that I'm saving 3D stuff,
			// how about a plane per 3rd dim, and separate save files per variable?
			std::shared_ptr<Image::ImageType<float>> image = std::make_shared<Image::ImageType<float>>(Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]), nullptr, 1, app.size.s[2]);

			int volume = getVolume();
			
			std::vector<real> stateVec(numStates() * volume);
			app.commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real) * numStates() * volume, &stateVec[0]);
			
			std::vector<real> potentialVec(volume);
			app.commands.enqueueReadBuffer(potentialBuffer, CL_TRUE, 0, sizeof(real) * volume, &potentialVec[0]);
			
			app.commands.finish();
			
			for (int channel = 0; channel < channelNames.size(); ++channel) {
				for (int z = 0; z < app.size.s[2]; ++z) {	
					for (int y = 0; y < app.size.s[1]; ++y) {
						for (int x = 0; x < app.size.s[0]; ++x) {
							int index = x + app.size.s[0] * (y + app.size.s[1] * z);
							real value = std::nan("");	
							if (channel < numStates()) {
								value = stateVec[channel + numStates() * index];
							} else {	//potential
								value = potentialVec[index];
							}
							(*image)(x,y,0,z) = value;
						}
					}
				}
				std::string filename = channelNames[channel] + std::to_string(i) + ".fits";
				std::cout << "saving file " << filename << std::endl;
				Image::system->write(filename, image); 
			}
			return;
		}
	}
}

void Solver3D::mouseMove(int x, int y, int dx, int dy) {
}

void Solver3D::mousePan(int dx, int dy) {
	switch (app.dim) {
	case 1:
	case 2:
		viewPos += Tensor::Vector<float,2>(-(float)dx * app.aspectRatio / (float)app.screenSize(0), (float)dy / (float)app.screenSize(1));
		break;
	case 3:
		{
			float magn = sqrt(dx * dx + dy * dy);
			float fdx = (float)dx / magn;
			float fdy = (float)dy / magn;
			Tensor::Quat<float> rotation = Tensor::Quat<float>(fdy, fdx, 0, magn * M_PI / 180.).fromAngleAxis();
			viewAngle = rotation * viewAngle;
			viewAngle /= Tensor::Quat<float>::length(viewAngle);
		}
		break;
	}
}

void Solver3D::mouseZoom(int dz) {
	switch (app.dim) {
	case 1:
	case 2:
		{
			float scale = exp((float)dz * -.03f);
			viewPos *= scale;
			viewZoom *= scale;
		}
		break;
	case 3:
		viewDist *= (float)exp((float)dz * -.03f);
		break;
	}
}

}
}


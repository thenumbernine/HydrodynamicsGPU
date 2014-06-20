#include "HydroGPU/Solver2D.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Image/FITS_IO.h"
#include "Common/File.h"
#include "Common/Exception.h"
#include <OpenGL/gl.h>
#include <iostream>

Solver2D::Solver2D(
	HydroGPUApp &app_,
	const std::string &programFilename)
: Super(app_, programFilename)
, fluidTex(GLuint())
, viewZoom(1.f)
{
	cl::Device device = app.device;
	cl::Context context = app.context;
	
	//memory

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];
	
	stateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume);
	
	if (app.dim == 1) {
		std::string shaderCode = Common::File::read("Display1D.shader");
		std::vector<Shader::Shader> shaders = {
			Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
			Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
		};
		shader1d = std::make_shared<Shader::Program>(shaders);
		shader1d->link();
		shader1d->setUniform<int>("tex", 0);
		shader1d->setUniform<float>("xmin", app.xmin.s[0]);
		shader1d->setUniform<float>("xmax", app.xmax.s[0]);
		shader1d->done();
	}
		
	//get a texture going for visualizing the output
	glGenTextures(1, &fluidTex);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	Tensor::Vector<int,3> glWraps(GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R);
	for (int i = 0; i < app.dim; ++i) {
		switch (app.boundaryMethods(i)) {
		case BOUNDARY_PERIODIC:
			glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_REPEAT);
			break;
		case BOUNDARY_MIRROR:
		case BOUNDARY_FREEFLOW:
			glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_CLAMP_TO_EDGE);
			break;
		}
	}
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, app.size.s[0], app.size.s[1], 0, GL_RGBA, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	int err = glGetError();
	if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;

	//hmm, my cl.hpp version only supports clCreateFromGLTexture2D, which is deprecated ... do I use the deprecated method, or do I stick with the C structures?
	// ... or do I look for a more up-to-date version of cl.hpp
	fluidTexMem = cl::ImageGL(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, fluidTex);

	addGravityKernel = cl::Kernel(program, "addGravity");
	app.setArgs(addGravityKernel, stateBuffer, gravityPotentialBuffer, app.size, app.dx, dtBuffer);

	convertToTexKernel = cl::Kernel(program, "convertToTex");
	app.setArgs(convertToTexKernel, stateBuffer, gravityPotentialBuffer, app.size, fluidTexMem, app.gradientTexMem);

	addDropKernel = cl::Kernel(program, "addDrop");
	app.setArgs(addDropKernel, stateBuffer, app.size, app.xmin, app.xmax);

	addSourceKernel = cl::Kernel(program, "addSource");
	app.setArgs(addSourceKernel, stateBuffer, app.size, app.xmin, app.xmax, dtBuffer);

	initKernels();	//parent call for after the child class has populated buffers
}

void Solver2D::resetState(std::vector<real8> state3DVec) {
	int volume = app.size.s[0] * app.size.s[1];
	if (volume != state3DVec.size()) throw Common::Exception() << "state vec is of bad size";

	//convert 3D to 2D
	std::vector<real4> stateVec;
	for (int i = 0; i < volume; ++i) {
		real8 &state3D = state3DVec[i];
		real4 state;
		state.s[0] = state3D.s[0];
		state.s[1] = state3D.s[1];
		state.s[2] = state3D.s[2];
		state.s[3] = state3D.s[4];
		stateVec.push_back(state);
	}
	if (volume != stateVec.size()) throw Common::Exception() << "state vec is of bad size";

	//grad^2 Phi = - 4 pi G rho
	//solve inverse discretized linear system to find Psi
	//D_ij / (-4 pi G) Phi_j = rho_i
	//once you get that, plug it into the total energy
	
	//write state density first for gravity potential, to then update energy
	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);
	
	//here's our initial guess to sor
	std::vector<real> gravityPotentialVec(volume);
	for (size_t i = 0; i < volume; ++i) {
		if (app.useGravity) {
			gravityPotentialVec[i] = stateVec[i].s[0];
		} else {
			gravityPotentialVec[i] = 0.;
		}
	}
	
	commands.enqueueWriteBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * volume, &gravityPotentialVec[0]);

	if (app.useGravity) {
		setPoissonRelaxRepeatArg();
		
		//solve for gravitational potential via gauss seidel
		cl::NDRange offsetNd(0, 0);
		for (int i = 0; i < 20; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offsetNd, globalSize, localSize);
		}

		//update internal energy
		for (int i = 0; i < volume; ++i) {
			stateVec[i].s[3] += gravityPotentialVec[i];
		}
	}
	
	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);
	commands.finish();
}

Solver2D::~Solver2D() {
	glDeleteTextures(1, &fluidTex);
	
	std::cout << "OpenCL profiling info:" << std::endl;
	for (EventProfileEntry *entry : entries) {
		std::cout << entry->name << "\t" 
			<< "duration " << entry->stat
			<< std::endl;
	}
}

void Solver2D::boundary() {
	//boundary
	for (int i = 0; i < app.dim; ++i) {
		cl::NDRange globalSize1d(app.size.s[i]);
		commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethods(i)][i], offset1d, globalSize1d, localSize1d);
	}
}

void Solver2D::initStep() {
}

void Solver2D::update() {
	//CPU need to bind beforehand for roe/cpu to use it
	//GPU needs it unbound until after the update
	if (!app.useGPU) {
		glBindTexture(GL_TEXTURE_2D, fluidTex);
	}
	
	Super::update();
	
	setPoissonRelaxRepeatArg();

	//commands.enqueueNDRangeKernel(addSourceKernel, offsetNd, globalSize, localSize, NULL, &addSourceEvent.clEvent);

	boundary();
	
	initStep();
	
	if (!app.useFixedDT) {
		calcTimestep();
	}
	
	step();

/*
	for (EventProfileEntry *entry : entries) {
		cl_ulong start = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong end = entry->clEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		entry->stat.accum((double)(end - start) * 1e-9);
	}
*/
}

void Solver2D::display() {
	glFinish();
	
	std::vector<cl::Memory> acquireGLMems = {fluidTexMem};
	commands.enqueueAcquireGLObjects(&acquireGLMems);

	if (app.useGPU) {
		convertToTexKernel.setArg(5, app.displayMethod);
		convertToTexKernel.setArg(6, app.displayScale);
		commands.enqueueNDRangeKernel(convertToTexKernel, offsetNd, globalSize, localSize);
	} else {
		int volume = app.size.s[0] * app.size.s[1];
		std::vector<real4> stateVec(volume);
		commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * volume, &stateVec[0]);  
		std::vector<Tensor::Vector<char,4>> texVec(volume);
		for (int i = 0; i < volume; ++i) {
			real value;
			switch (app.displayMethod) {
			case DISPLAY_DENSITY:	//density
				value = stateVec[i].s[0];
				break;
			case DISPLAY_VELOCITY:	//velocity
				value = sqrt(stateVec[i].s[1] * stateVec[i].s[1] + stateVec[i].s[2] * stateVec[i].s[2]) / stateVec[i].s[0];
				break;
			case DISPLAY_PRESSURE:	//pressure
				value = (app.gamma - 1.f) * stateVec[i].s[3] * stateVec[i].s[0];
				break;
			default:
				value = .5f;
				break;
			}		
			texVec[i](0) = (char)(255.f * value * app.displayScale);
		}
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.size.s[0], app.size.s[1], GL_RGBA, GL_UNSIGNED_BYTE, &texVec[0].v);
	}

	commands.enqueueReleaseGLObjects(&acquireGLMems);
	commands.finish();
	
	glPushMatrix();
	glTranslatef(-viewPos(0), -viewPos(1), 0);
	glScalef(viewZoom, viewZoom, viewZoom);
	
	if (app.dim == 1) {

		shader1d->use();
		glBindTexture(GL_TEXTURE_2D, fluidTex);
		glBegin(GL_LINE_STRIP);
		for (int i = 2; i < app.size.s[0]-2; ++i) {
			real x = ((real)(i) + .5f) / (real)app.size.s[0];
			glVertex2f(x, 0.f);
		}
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);	
		shader1d->done();
		
		glBegin(GL_LINES);
		glColor3f(.5, .5, .5);
		glVertex2f(0, -10);
		glVertex2f(0, 10);
		glVertex2f(-10, 0);
		glVertex2f(10, 0);
		glColor3f(.25, .25, .25);
		for (int i = -100; i < 100; ++i) {
			glVertex2f(.1*i, -10);
			glVertex2f(.1*i, 10);
			glVertex2f(-10, .1*i);
			glVertex2f(10, .1*i);
		}
		glEnd();
		
	} else {

		glBindTexture(GL_TEXTURE_2D, fluidTex);
		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
		glTexCoord2f(0,0); glVertex2f(app.xmin.s[0], app.xmin.s[1]);
		glTexCoord2f(1,0); glVertex2f(app.xmax.s[0], app.xmin.s[1]);
		glTexCoord2f(1,1); glVertex2f(app.xmax.s[0], app.xmax.s[1]);
		glTexCoord2f(0,1); glVertex2f(app.xmin.s[0], app.xmax.s[1]);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glPopMatrix();

	if (app.showTimestep) {
		real dt;
		commands.enqueueReadBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &dt);
		commands.finish();
		std::cout << "dt " << dt << std::endl;
	}
	
	{int err = glGetError();
	if (err) std::cout << "GL error " << err << " at " << __LINE__ << std::endl;}
}

void Solver2D::resize() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-app.aspectRatio *.5, app.aspectRatio * .5, -.5, .5, -1., 1.);
	glMatrixMode(GL_MODELVIEW);
}

void Solver2D::addDrop() {
	addSourcePos.s[0] = mousePos(0);
	addSourcePos.s[1] = mousePos(1);
	addSourceVel.s[0] = mouseVel(0);
	addSourceVel.s[1] = mouseVel(1);
	addDropKernel.setArg(4, addSourcePos);
	addDropKernel.setArg(5, addSourceVel);
	commands.enqueueNDRangeKernel(addDropKernel, offsetNd, globalSize, localSize);
}

void Solver2D::screenshot() {
	for (int i = 0; i < 1000; ++i) {
		std::string filename = std::string("screenshot") + std::to_string(i) + ".png";
		if (!Common::File::exists(filename)) {
			std::shared_ptr<Image::Image> image = std::make_shared<Image::Image>(
				Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]),
				nullptr, 3);
			
			glBindTexture(GL_TEXTURE_2D, fluidTex);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, image->getData());
			glBindTexture(GL_TEXTURE_2D, 0);
			
			Image::system->write(filename, image);
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

void Solver2D::save() {
	for (int i = 0; i < 1000; ++i) {
		std::string filename = std::string("save") + std::to_string(i) + ".fits";
		if (!Common::File::exists(filename)) {
			save(filename);
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

void Solver2D::save(std::string filename) {
	std::shared_ptr<Image::ImageType<float>> image = std::make_shared<Image::ImageType<float>>(Tensor::Vector<int,2>(app.size.s[0], app.size.s[1]), nullptr, 1, 5);
	
	std::vector<real4> stateVec(app.size.s[0] * app.size.s[1]);
	app.commands.enqueueReadBuffer(stateBuffer, CL_TRUE, 0, sizeof(real4) * stateVec.size(), &stateVec[0]);
		
	std::vector<real> gravVec(app.size.s[0] * app.size.s[1]);
	app.commands.enqueueReadBuffer(gravityPotentialBuffer, CL_TRUE, 0, sizeof(real) * gravVec.size(), &gravVec[0]);
	
	app.commands.finish();
	
	for (int j = 0; j < app.size.s[1]; ++j) {
		for (int i = 0; i < app.size.s[0]; ++i) {
			real4 *state = &stateVec[i + app.size.s[0] * j];
			real grav = gravVec[i + app.size.s[0] * j];
			(*image)(i,j,0,0) = state->s[0];
			(*image)(i,j,0,1) = state->s[1] / state->s[0];
			(*image)(i,j,0,2) = state->s[2] / state->s[0];
			(*image)(i,j,0,3) = state->s[3] / state->s[0];
			(*image)(i,j,0,4) = grav;
		}
	}
	Image::system->write(filename, image); 
}

void Solver2D::mouseMove(int x, int y, int dx, int dy) {
	mousePos(0) = (float)x / (float)app.screenSize(0) * (app.xmax.s[0] - app.xmin.s[0]) + app.xmin.s[0];
	mousePos(0) *= app.aspectRatio;	//only if xmin/xmax is symmetric. otehrwise more math required.
	mousePos(1) = (1.f - (float)y / (float)app.screenSize(1)) * (app.xmax.s[1] - app.xmin.s[1]) + app.xmin.s[1];
	mousePos += viewPos;
	mousePos /= viewZoom;
	mouseVel(0) = (float)dx / (float)app.screenSize(0);
	mouseVel(1) = (float)dy / (float)app.screenSize(1);
}

void Solver2D::mousePan(int dx, int dy) {
	viewPos += Tensor::Vector<float,2>(-(float)dx * app.aspectRatio / (float)app.screenSize(0), (float)dy / (float)app.screenSize(1));
}

void Solver2D::mouseZoom(int dz) {
	float scale = exp((float)dz * -.03f); 
	viewPos *= scale;
	viewZoom *= scale; 
}



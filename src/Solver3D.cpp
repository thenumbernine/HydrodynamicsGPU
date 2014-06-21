#include "HydroGPU/Solver3D.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Image/FITS_IO.h"
#include "Common/File.h"
#include "Common/Exception.h"
#include <OpenGL/gl.h>
#include <iostream>

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

#if 0
template<typename T, int n> struct CLType;

#define DEFINE_CL_PRIM_TEMPLATE(type,cltypesuffix)	\
/*template<> struct CLType<type,1> { typedef cl_##cltypesuffix Type; };*/	\
template<> struct CLType<type,2> { typedef cl_##cltypesuffix##2 Type; };	\
template<> struct CLType<type,4> { typedef cl_##cltypesuffix##4 Type; };	\
template<> struct CLType<type,8> { typedef cl_##cltypesuffix##8 Type; };	\
template<> struct CLType<type,16> { typedef cl_##cltypesuffix##16 Type; };

#define DEFINE_CL_PRIM_TEMPLATE_SU(type)	\
DEFINE_CL_PRIM_TEMPLATE(type, type)	\
DEFINE_CL_PRIM_TEMPLATE(unsigned type, u##type)

DEFINE_CL_PRIM_TEMPLATE_SU(char)
DEFINE_CL_PRIM_TEMPLATE_SU(short)
DEFINE_CL_PRIM_TEMPLATE_SU(int)
DEFINE_CL_PRIM_TEMPLATE_SU(long)
DEFINE_CL_PRIM_TEMPLATE(float, float)

template<typename T, int n>
std::ostream& operator<<(std::ostream& o, typename CLType<T,n>::Type v) {
	const char* comma = "";
	for (int i = 0; i < n; ++i) {
		o << comma << v.s[i];
		comma = ", ";
	}
	return o;
}
#endif

Solver3D::Solver3D(
	HydroGPUApp &app_,
	const std::string &programFilename)
: Super(app_, std::vector<std::string>{"Common3D.cl", programFilename})
, fluidTex(GLuint())
, viewZoom(1.f)
, viewDist(1.f)
{
	cl::Context context = app.context;
	

	//memory

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];
	
	stateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume);

	switch (app.dim) {
	case 1:
		{
			std::string shaderCode = Common::File::read("Display1D.shader");
			std::vector<Shader::Shader> shaders = {
				Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
				Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
			};
			shader = std::make_shared<Shader::Program>(shaders);
			shader->link();
			shader->setUniform<int>("tex", 0);
			shader->setUniform<float>("xmin", app.xmin.s[0]);
			shader->setUniform<float>("xmax", app.xmax.s[0]);
			shader->done();
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
		}
		break;
	case 3:
		{
			std::string shaderCode = Common::File::read("Display3D.shader");
			std::vector<Shader::Shader> shaders = {
				Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
				Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
			};
			shader = std::make_shared<Shader::Program>(shaders);
			shader->link()
				.setUniform<int>("tex", 0)
				.setUniform<int>("maxiter", 100)
				.setUniform<float>("scale", app.xmax.s[0] - app.xmin.s[0], app.xmax.s[1] - app.xmin.s[1], app.xmax.s[2] - app.xmin.s[2])
				.done();
		
			//get a texture going for visualizing the output
			glGenTextures(1, &fluidTex);
			glBindTexture(GL_TEXTURE_3D, fluidTex);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
			glTexImage3D(GL_TEXTURE_3D, 0, 4, app.size.s[0], app.size.s[1], app.size.s[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glBindTexture(GL_TEXTURE_3D, 0);
			int err = glGetError();
			if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;
		}
		break;
	}

	fluidTexMem = cl::ImageGL(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_3D, 0, fluidTex);
	
	convertToTexKernel = cl::Kernel(program, "convertToTex");
	app.setArgs(convertToTexKernel, stateBuffer, gravityPotentialBuffer, app.size, fluidTexMem, app.gradientTexMem);

	initKernels();
}

void Solver3D::resetState(std::vector<real8> stateVec) {	
	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];
	if (volume != stateVec.size()) throw Common::Exception() << "got a state vec with a bad length";

	//grad^2 Phi = - 4 pi G rho
	//solve inverse discretized linear system to find Psi
	//D_ij / (-4 pi G) Phi_j = rho_i
	//once you get that, plug it into the total energy
	
	//write state density first for gravity potential, to then update energy
	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real8) * volume, &stateVec[0]);
	
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
		for (int i = 0; i < 20; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offsetNd, globalSize, localSize);
		}

		//update internal energy
		for (int i = 0; i < volume; ++i) {
			stateVec[i].s[4] += gravityPotentialVec[i];
		}
	}
	
	commands.enqueueWriteBuffer(stateBuffer, CL_TRUE, 0, sizeof(real8) * volume, &stateVec[0]);
	commands.finish();
}

Solver3D::~Solver3D() {
	glDeleteTextures(1, &fluidTex);
	
	std::cout << "OpenCL profiling info:" << std::endl;
	for (EventProfileEntry *entry : entries) {
		std::cout << entry->name << "\t" 
			<< "duration " << entry->stat
			<< std::endl;
	}
}

void Solver3D::boundary() {
	switch (app.dim) {
	case 1:
	case 2:
		//boundary
		for (int i = 0; i < app.dim; ++i) {
			cl::NDRange globalSize1d(app.size.s[i]);
			commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethods(i)][i], offset1d, globalSize1d, localSize1d);
		}
		break;
	case 3:
		//boundary
		cl::NDRange offset2d(0, 0);
		cl::NDRange localSize2d(localSize[0], localSize[1]);
		for (int i = 0; i < app.dim; ++i) {
			cl::NDRange globalSize2d;
			switch (i) {
			case 0:
				globalSize2d = cl::NDRange(app.size.s[0], app.size.s[1]);
				break;
			case 1:
				globalSize2d = cl::NDRange(app.size.s[0], app.size.s[2]);
				break;
			case 2:
				globalSize2d = cl::NDRange(app.size.s[1], app.size.s[2]);
				break;
			}
			commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethods(i)][i], offset2d, globalSize2d, localSize2d);
		}
		break;
	}
}

void Solver3D::display() {
	glFinish();
	
	std::vector<cl::Memory> acquireGLMems = {fluidTexMem};
	commands.enqueueAcquireGLObjects(&acquireGLMems);

	if (app.useGPU) {
		convertToTexKernel.setArg(5, app.displayMethod);
		convertToTexKernel.setArg(6, app.displayScale);
		commands.enqueueNDRangeKernel(convertToTexKernel, offsetNd, globalSize, localSize);
	} else {
		throw Common::Exception() << "no support";
	}

	commands.enqueueReleaseGLObjects(&acquireGLMems);
	commands.finish();

	switch (app.dim) {
	case 1:
		glPushMatrix();
		glTranslatef(-viewPos(0), -viewPos(1), 0);
		glScalef(viewZoom, viewZoom, viewZoom);
		
		glBindTexture(GL_TEXTURE_2D, fluidTex);
		shader->use();
		for (int channel = 0; channel < 3; ++channel) {
			shader->setUniform<int>("channel", channel);
			glBegin(GL_LINE_STRIP);
			for (int i = 2; i < app.size.s[0]-2; ++i) {
				real x = ((real)(i) + .5f) / (real)app.size.s[0];
				glVertex2f(x, 0.f);
			}
			glEnd();
		}
		shader->done();
		glBindTexture(GL_TEXTURE_2D, 0);	
		
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
	
		glPopMatrix();
			
		break;
	case 2:
		glPushMatrix();
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
	
		glPopMatrix();
	
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
					shader->use();
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
					shader->done();
					glDisable(GL_BLEND);
					glDisable(GL_DEPTH_TEST);
					glCullFace(GL_BACK);
					glDisable(GL_CULL_FACE);
				}
			}
		}
		break;
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
}

void Solver3D::save() {
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


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
: Super(app_, programFilename)
, fluidTex(GLuint())
, viewDist(1.f)
{
	cl::Device device = app.device;
	cl::Context context = app.context;
	

	//memory

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];
	
	stateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume);
	
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
	}

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

	fluidTexMem = cl::ImageGL(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_3D, 0, fluidTex);
		
	addGravityKernel = cl::Kernel(program, "addGravity");
	app.setArgs(addGravityKernel, stateBuffer, gravityPotentialBuffer, app.size, app.dx, dtBuffer);

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
	//boundary
	for (int i = 0; i < 3; ++i) {
		cl::NDRange globalSize2d(app.size.s[(i+1)%3], app.size.s[(i+2)%3]);
		cl::NDRange offset2d(0, 0);
		cl::NDRange localSize2d(localSize[(i+1)%3], localSize[(i+2)%3]);
		commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethods(i)][i], offset2d, globalSize2d, localSize2d);
	}
}


void Solver3D::initStep() {
}

void Solver3D::update() {
	Super::update();
	
	setPoissonRelaxRepeatArg();

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
	
	{int err = glGetError();
	if (err) std::cout << "GL error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl;}
}

void Solver3D::resize() {
	const float zNear = .01;
	const float zFar = 10;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-app.aspectRatio * zNear, app.aspectRatio * zNear, -zNear, zNear, zNear, zFar);
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
	float magn = sqrt(dx * dx + dy * dy);
	float fdx = (float)dx / magn;
	float fdy = (float)dy / magn;
	Tensor::Quat<float> rotation = Tensor::Quat<float>(fdy, fdx, 0, magn * M_PI / 180.).fromAngleAxis();
	viewAngle = rotation * viewAngle;
	viewAngle /= Tensor::Quat<float>::length(viewAngle);

}

void Solver3D::mouseZoom(int dz) {
		viewDist *= (float)exp((float)dz * -.03f);
}


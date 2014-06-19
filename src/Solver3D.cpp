#include "HydroGPU/Solver3D.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Image/FITS_IO.h"
#include "Common/File.h"
#include "Common/Exception.h"
#include <OpenGL/gl.h>
#include <iostream>

const int DIM = 3;

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
: app(app_)
, commands(app.commands)
, fluidTex(GLuint())
, viewDist(1.f)
{
	cl::Device device = app.device;
	cl::Context context = app.context;
		
	stateBoundaryKernels.resize(NUM_BOUNDARY_METHODS);
	for (std::vector<cl::Kernel>& v : stateBoundaryKernels) {
		v.resize(DIM);
	}
	
	{
		std::string shaderCode = Common::File::read("display.shader");
		std::vector<Shader::Shader> shaders = {
			Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", shaderCode}),
			Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", shaderCode})
		};
		shader = std::make_shared<Shader::Program>(shaders);
		shader->link();
		shader->setUniform<int>("tex", 0);
		shader->setUniform<int>("maxiter", 100);
		shader->setUniform<float>("scale", app.xmax.s[0] - app.xmin.s[0], app.xmax.s[1] - app.xmin.s[1], app.xmax.s[2] - app.xmin.s[2]);
		shader->done();
	}

	//get a texture going for visualizing the output
	glGenTextures(1, &fluidTex);
	glBindTexture(GL_TEXTURE_3D, fluidTex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexImage3D(GL_TEXTURE_3D, 0, 4, app.size.s[0], app.size.s[1], app.size.s[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_3D, 0);
	int err = glGetError();
	if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;

	fluidTexMem = cl::ImageGL(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_3D, 0, fluidTex);

	size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	Tensor::Vector<size_t,DIM> localSizeVec;
	for (int n = 0; n < DIM; ++n) {
		localSizeVec(n) = std::min<size_t>(maxWorkItemSizes[n], app.size.s[n]);
	}
	while (localSizeVec.volume() > maxWorkGroupSize) {
		for (int n = 0; n < DIM; ++n) {
			localSizeVec(n) = (size_t)ceil((double)localSizeVec(n) * .5);
		}
	}
	
	//hmm...
	if (!app.useGPU) localSizeVec(0) >>= 1;
	std::cout << "global_size\t" << app.size << std::endl;
	std::cout << "local_size\t" << localSizeVec << std::endl;

	globalSize = cl::NDRange(app.size.s[0], app.size.s[1], app.size.s[2]);
	localSize = cl::NDRange(localSizeVec(0), localSizeVec(1), localSizeVec(2));
	localSize1d = cl::NDRange(localSizeVec(0));
	offset1d = cl::NDRange(0);
	offset3d = cl::NDRange(0, 0, 0);

	{
		std::vector<std::string> kernelSources = std::vector<std::string>{
			std::string() + "#define GAMMA " + std::to_string(app.gamma) + "f\n",
			Common::File::read("Common3D.cl"),
			Common::File::read(programFilename)
		};
		std::vector<std::pair<const char *, size_t>> sources;
		for (const std::string &s : kernelSources) {
			sources.push_back(std::pair<const char *, size_t>(s.c_str(), s.length()));
		}
		program = cl::Program(context, sources);
	}

	try {
		program.build({device}, "-I include");
	} catch (cl::Error &err) {
		throw Common::Exception() 
			<< "failed to build program executable!\n"
			<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	}

	//warnings?
	std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;

	//memory

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];

	stateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume);
	cflBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	cflSwapBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume / localSize[0]);
	dtBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real));
	gravityPotentialBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);

	//get the edges, so reduction doesn't
	{
		std::vector<real> cflVec(volume);
		for (real &r : cflVec) { r = std::numeric_limits<real>::max(); }
		commands.enqueueWriteBuffer(cflBuffer, CL_TRUE, 0, sizeof(real) * volume, &cflVec[0]);
	}
	
	if (app.useFixedDT) {
		commands.enqueueWriteBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &app.fixedDT);
	}

	for (int boundaryIndex = 0; boundaryIndex < NUM_BOUNDARY_METHODS; ++boundaryIndex) {
		for (int side = 0; side < DIM; ++side) {
			std::string name = "stateBoundary";
			
			switch (boundaryIndex) {
			case BOUNDARY_PERIODIC:
				name += "Periodic";
				break;
			case BOUNDARY_MIRROR:
				name += "Mirror";
				break;
			case BOUNDARY_FREEFLOW:
				name += "FreeFlow";
				break;
			default:
				throw Common::Exception() << "no kernel for boundary method " << boundaryIndex;
			}
			name += side == 0 ? "X" : (side == 1 ? "Y" : "Z");
			stateBoundaryKernels[boundaryIndex][side] = cl::Kernel(program, name.c_str());
			app.setArgs(stateBoundaryKernels[boundaryIndex][side], stateBuffer, app.size);
		}
	}

	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	app.setArgs(calcCFLMinReduceKernel, cflBuffer, cl::Local(localSizeVec(0) * sizeof(real)), volume, cflSwapBuffer);

	poissonRelaxKernel = cl::Kernel(program, "poissonRelax");
	app.setArgs(poissonRelaxKernel, gravityPotentialBuffer, stateBuffer, app.size, app.dx);
	
	addGravityKernel = cl::Kernel(program, "addGravity");
	app.setArgs(addGravityKernel, stateBuffer, gravityPotentialBuffer, app.size, app.dx, dtBuffer);

	convertToTexKernel = cl::Kernel(program, "convertToTex");
	app.setArgs(convertToTexKernel, stateBuffer, gravityPotentialBuffer, app.size, fluidTexMem, app.gradientTexMem);
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
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset3d, globalSize, localSize);
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
	for (int i = 0; i < DIM; ++i) {
		cl::NDRange globalSize2d(app.size.s[(i+1)%DIM], app.size.s[(i+2)%DIM]);
		cl::NDRange offset2d(0, 0);
		cl::NDRange localSize2d(localSize[(i+1)%DIM], localSize[(i+2)%DIM]);
		commands.enqueueNDRangeKernel(stateBoundaryKernels[app.boundaryMethods(i)][i], offset2d, globalSize2d, localSize2d);
	}
}

void Solver3D::setPoissonRelaxRepeatArg() {
	cl_int3 repeat;
	for (int i = 0; i < 3; ++i) {
		switch (app.boundaryMethods(0)) {	//TODO per dimension
		case BOUNDARY_PERIODIC:
			repeat.s[i] = 1;
			break;
		case BOUNDARY_MIRROR:
		case BOUNDARY_FREEFLOW:
			repeat.s[i] = 0;
			break;
		default:
			throw Common::Exception() << "unknown boundary method " << app.boundaryMethods(0);
		}	
	}	
	poissonRelaxKernel.setArg(4, repeat);
}

void Solver3D::initStep() {
}

void Solver3D::findMinTimestep() {
	int reduceSize = globalSize[0] * globalSize[1] * globalSize[2];
	cl::Buffer dst = cflSwapBuffer;
	cl::Buffer src = cflBuffer;
	while (reduceSize > 1) {
		int nextSize = (reduceSize >> 4) + !!(reduceSize & ((1 << 4) - 1));
		cl::NDRange reduceGlobalSize(std::max<int>(reduceSize, localSize[0]));
		calcCFLMinReduceKernel.setArg(0, src);
		calcCFLMinReduceKernel.setArg(2, reduceSize);
		calcCFLMinReduceKernel.setArg(3, nextSize == 1 ? dtBuffer : dst);
		commands.enqueueNDRangeKernel(calcCFLMinReduceKernel, offset1d, reduceGlobalSize, localSize1d);
		commands.finish();
		std::swap(dst, src);
		reduceSize = nextSize;
	}
}

void Solver3D::update() {
	setPoissonRelaxRepeatArg();

	boundary();
	
	initStep();
	
	if (!app.useFixedDT) {
		calcTimestep();
	}

	if (false) {
		real dt;
		commands.enqueueReadBuffer(dtBuffer, CL_TRUE, 0, sizeof(real), &dt);
		std::cout << "dt " << dt << std::endl;
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
		commands.enqueueNDRangeKernel(convertToTexKernel, offset3d, globalSize, localSize);
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


#include "HydroGPU/RoeSolver.h"
#include "Common/Exception.h"
#include "Common/Finally.h"
#include "Common/Macros.h"
#include "TensorMath/Vector.h"
#include <OpenGL/gl.h>
#include <fstream>

std::string readFile(std::string filename) {
	std::ifstream f(filename);
	f.seekg(0, f.end);
	size_t len = f.tellg();
	f.seekg(0, f.beg);
	char *buf = new char[len];
	Finally finally([&](){ delete[] buf; });
	f.read(buf, len);
	return std::string(buf, len);
}

RoeSolver::RoeSolver(
	cl::Device device,
	cl::Context context,
	cl_int2 size_,
	cl::CommandQueue commands_,
	std::vector<Cell> &cells,
	real* xmin,
	real* xmax,
	cl_mem fluidTexMem,
	cl_mem gradientTexMem,
	bool useGPU_)
: commands(commands_)
, useGPU(useGPU_)
, cfl(.5f)
{
	size = size_;

	size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	Vector<size_t,DIM> globalSizeVec, localSizeVec;
	for (int n = 0; n < DIM; ++n) {
		globalSizeVec(n) = size.s[n];
		localSizeVec(n) = std::min<size_t>(maxWorkItemSizes[n], size.s[n]);
	}
	while (localSizeVec.volume() > maxWorkGroupSize) {
		for (int n = 0; n < DIM; ++n) {
			localSizeVec(n) = (size_t)ceil((double)localSizeVec(n) * .5);
		}
	}
	//hmm...
	if (!useGPU) localSizeVec(0) >>= 1;
	std::cout << "global_size\t" << globalSizeVec << std::endl;
	std::cout << "local_size\t" << localSizeVec << std::endl;

	globalSize = cl::NDRange(globalSizeVec(0), globalSizeVec(1));
	localSize = cl::NDRange(localSizeVec(0), localSizeVec(1));

	std::string kernelSource = readFile("res/roe_euler_2d.cl");
	std::vector<std::pair<const char *, size_t>> sources = {
		std::pair<const char *, size_t>(kernelSource.c_str(), kernelSource.length())
	};
	program = cl::Program(context, sources);
 
	try {
		program.build({device}, "-I res/include");
	} catch (cl::Error &err) {
		std::cout << "failed to build program executable!" << std::endl;
		
		std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
		std::cout << log << std::endl;
		exit(1);
	}
 
	unsigned int count = size.s[0] * size.s[1];
	cellsMem = cl::Buffer(context,  CL_MEM_READ_WRITE, sizeof(Cell) * count);
	commands.enqueueWriteBuffer(cellsMem, CL_TRUE, 0, sizeof(Cell) * count, &cells[0]);
	
	cflMem = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * count);
	cflTimestepMem = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real));

	calcEigenDecompositionKernel = cl::Kernel(program, "calcEigenDecomposition");
	calcCFLAndDeltaQTildeKernel = cl::Kernel(program, "calcCFLAndDeltaQTilde");
	calcCFLMinReduceKernel = cl::Kernel(program, "calcCFLMinReduce");
	calcCFLMinFinalKernel = cl::Kernel(program, "calcCFLMinFinal");
	calcRTildeKernel = cl::Kernel(program, "calcRTilde");
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	updateStateKernel = cl::Kernel(program, "updateState");
	convertToTexKernel = cl::Kernel(program, "convertToTex");
	addDropKernel = cl::Kernel(program, "addDrop");

	std::vector<cl::Kernel*> kernels = {
		&calcEigenDecompositionKernel,
		&calcCFLAndDeltaQTildeKernel,
		&calcRTildeKernel,
		&calcFluxKernel,
		&updateStateKernel,
		&addDropKernel,
	};
	std::for_each(kernels.begin(), kernels.end(), [&](cl::Kernel* kernel) {
		kernel->setArg(0, cellsMem);
		kernel->setArg(1, size);
	});

	real2 dx;
	for (int i = 0; i < DIM; ++i) {
		dx.s[i] = (xmax[i] - xmin[i]) / (float)size.s[i];
	}
	
	calcFluxKernel.setArg(2, dx);
	calcFluxKernel.setArg(3, cflTimestepMem);

	updateStateKernel.setArg(2, dx);
	updateStateKernel.setArg(3, cflTimestepMem);
	
	calcCFLAndDeltaQTildeKernel.setArg(2, cflMem);
	calcCFLAndDeltaQTildeKernel.setArg(3, dx);

	calcCFLMinReduceKernel.setArg(0, cflMem);
	calcCFLMinReduceKernel.setArg(1, cl::__local(localSizeVec(0) * sizeof(real)));
	
	calcCFLMinFinalKernel.setArg(0, cflMem);
	calcCFLMinFinalKernel.setArg(1, cl::__local(localSizeVec(0) * sizeof(real)));
	calcCFLMinFinalKernel.setArg(2, cflTimestepMem);
	calcCFLMinFinalKernel.setArg(3, cfl);

	addDropKernel.setArg(2, cflTimestepMem);

	//if (useGPU) 
	{
		convertToTexKernel.setArg(0, cellsMem);
		convertToTexKernel.setArg(1, size);
		convertToTexKernel.setArg(2, fluidTexMem);
		convertToTexKernel.setArg(3, gradientTexMem);
	}
}

void RoeSolver::update(cl_mem fluidTexMem) {
	cl::NDRange offset2d(0, 0);

	commands.enqueueNDRangeKernel(calcEigenDecompositionKernel, offset2d, globalSize, localSize);
	commands.enqueueNDRangeKernel(calcCFLAndDeltaQTildeKernel, offset2d, globalSize, localSize);

	{
		cl::NDRange offset1d(0);
		cl::NDRange reduceGlobalSize(globalSize[0] * globalSize[1] / 4);
		cl::NDRange reduceLocalSize(localSize[0]);
		commands.enqueueNDRangeKernel(calcCFLMinReduceKernel, offset1d, reduceGlobalSize, reduceLocalSize);
	
		while (reduceGlobalSize[0] / localSize[0] > localSize[0]) {
			reduceGlobalSize = cl::NDRange(reduceGlobalSize[0] / localSize[0]);
			commands.enqueueNDRangeKernel(calcCFLMinReduceKernel, offset1d, reduceGlobalSize, reduceLocalSize);
		}
		reduceGlobalSize = cl::NDRange(reduceGlobalSize[0] / localSize[0]);

		calcCFLMinFinalKernel.setArg(4, reduceGlobalSize);
		commands.enqueueNDRangeKernel(calcCFLMinFinalKernel, offset1d, reduceLocalSize, reduceLocalSize);
	}

	commands.enqueueNDRangeKernel(calcRTildeKernel, offset2d, globalSize, localSize);
	commands.enqueueNDRangeKernel(calcFluxKernel, offset2d, globalSize, localSize);
	commands.enqueueNDRangeKernel(updateStateKernel, offset2d, globalSize, localSize);

	glFlush();
	glFinish();
	clEnqueueAcquireGLObjects(commands(), 1, &fluidTexMem, 0, 0, 0);

	if (useGPU) {
		commands.enqueueNDRangeKernel(convertToTexKernel, offset2d, globalSize, localSize);
	} else {
		int count = size.s[0] * size.s[1];
		std::vector<Cell> cells(count);
		commands.enqueueReadBuffer(cellsMem, CL_TRUE, 0, sizeof(Cell) * count, &cells[0]);  
		std::vector<Vector<char,4>> buffer(count);
		for (int i = 0; i < count; ++i) {
			buffer[i](0) = (char)(255.f * cells[i].q.s[0] * .9f);
		}
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.s[0], size.s[1], GL_RGBA, GL_UNSIGNED_BYTE, &buffer[0].v);
	}

	clEnqueueReleaseGLObjects(commands(), 1, &fluidTexMem, 0, 0, 0);
	commands.flush();
	commands.finish();
}

void RoeSolver::addDrop(Vector<float,2> pos, Vector<float,2> vel) {
	addSourcePos.s[0] = pos(0);
	addSourcePos.s[1] = pos(1);
	addSourceVel.s[0] = vel(0);
	addSourceVel.s[1] = vel(1);
	addDropKernel.setArg(3, addSourcePos);
	addDropKernel.setArg(4, addSourceVel);
	commands.enqueueNDRangeKernel(addDropKernel, cl::NDRange(0,0), globalSize, localSize);
}


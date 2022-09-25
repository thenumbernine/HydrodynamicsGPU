#include "HydroGPU/Plot/VectorField.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Plot {

VectorField::VectorField(std::shared_ptr<HydroGPU::Solver::Solver> solver_, int resolution_)
: solver(solver_)
, resolution(resolution_)
{
	//create GL buffer
	int volume = 1;
	for (int i = 0; i < solver->app->dim; ++i) {
		volume *= resolution;
	}
	vertexCount = 3 * 6 * volume;
	glBuffer = GLCxx::ArrayBuffer(sizeof(float) * vertexCount, nullptr, GL_DYNAMIC_DRAW);
	solver->cl.totalAlloc += sizeof(float) * vertexCount;
	
	//create CL interop
	if (solver->app->hasGLSharing) {
		vertexBufferGL = cl::BufferGL(solver->app->clCommon->context, CL_MEM_READ_WRITE, glBuffer());
	} else {
		vertexBufferCL = solver->cl.alloc(sizeof(float) * vertexCount);
		vertexBufferCPU.resize(vertexCount);
	}
	
	//create transfer kernel
	updateVectorFieldKernel = cl::Kernel(solver->program, "updateVectorField");
	if (solver->app->hasGLSharing) {
		updateVectorFieldKernel.setArg(0, vertexBufferGL);
	} else {
		updateVectorFieldKernel.setArg(0, vertexBufferCL);
	}
	updateVectorFieldKernel.setArg(1, scale);
	updateVectorFieldKernel.setArg(2, variable);
}

void VectorField::display() {
	//glFlush();
	cl::NDRange global;
	switch (solver->app->dim) {
	case 1:
		global = cl::NDRange(resolution);
		break;
	case 2:
		global = cl::NDRange(resolution, resolution);
		break;
	case 3:
		global = cl::NDRange(resolution, resolution, resolution);
		break;
	}
	updateVectorFieldKernel.setArg(1, scale);
	updateVectorFieldKernel.setArg(2, variable);	//equation->vectorFieldVars
	solver->equation->setupUpdateVectorFieldKernelArgs(updateVectorFieldKernel, solver.get());
	
	solver->app->clCommon->commands.enqueueNDRangeKernel(updateVectorFieldKernel, solver->offsetNd, global, solver->localSize);
	solver->app->clCommon->commands.finish();

	if (!solver->app->hasGLSharing) {
		solver->app->clCommon->commands.enqueueReadBuffer(vertexBufferCL, CL_TRUE, 0, sizeof(float) * vertexCount, vertexBufferCPU.data());
		glBuffer.updateData(sizeof(float) * vertexCount, vertexBufferCPU.data());
	}

	glBuffer.bind();
	glColor3f(1,1,1);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_LINES, 0, vertexCount);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBuffer.unbind();

}

}
}

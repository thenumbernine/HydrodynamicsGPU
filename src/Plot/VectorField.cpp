#include "HydroGPU/Plot/VectorField.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Plot {

VectorField::VectorField(std::shared_ptr<HydroGPU::Solver::Solver> solver_, int resolution_)
: solver(solver_)
, glBuffer(0)
, resolution(resolution_)
, vertexCount(0)
, variable(0)
, scale(.125f)
{
	//create GL buffer
	glGenBuffers(1, &glBuffer);
	glBindBuffer(GL_ARRAY_BUFFER_ARB, glBuffer);
	int volume = 1;
	for (int i = 0; i < solver->app->dim; ++i) {
		volume *= resolution;
	}
	vertexCount = 3 * 6 * volume;
	glBufferData(GL_ARRAY_BUFFER_ARB, sizeof(float) * vertexCount, nullptr, GL_DYNAMIC_DRAW_ARB);
	solver->cl.totalAlloc += sizeof(float) * vertexCount;
	glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	//create CL interop
	vertexBuffer = cl::BufferGL(solver->app->clCommon->context, CL_MEM_READ_WRITE, glBuffer);
	//create transfer kernel
	updateVectorFieldKernel = cl::Kernel(solver->program, "updateVectorField");
	updateVectorFieldKernel.setArg(0, vertexBuffer);
	updateVectorFieldKernel.setArg(1, scale);
	updateVectorFieldKernel.setArg(2, variable);
}

VectorField::~VectorField() {
	glDeleteBuffers(1, &glBuffer);	
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
	updateVectorFieldKernel.setArg(1, (real)scale);
	updateVectorFieldKernel.setArg(2, variable);	//equation->vectorFieldVars
	solver->equation->setupUpdateVectorFieldKernelArgs(updateVectorFieldKernel, solver.get());
	
	solver->app->clCommon->commands.enqueueNDRangeKernel(updateVectorFieldKernel, solver->offsetNd, global, solver->localSize);
	solver->app->clCommon->commands.finish();

	glDisable(GL_DEPTH_TEST);

	glBindBuffer(GL_ARRAY_BUFFER_ARB, glBuffer);
	glColor3d(1,1,1);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_LINES, 0, vertexCount);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	
	glEnable(GL_DEPTH_TEST);
}

}
}

#include "HydroGPU/Plot/VectorField.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Plot {

VectorField::VectorField(std::shared_ptr<HydroGPU::Solver::Solver> solver_)
: solver(solver_)
, vectorFieldGLBuffer(0)
, vectorFieldResolution(0)
, vectorFieldVertexCount(0)
{
	vectorFieldResolution = 16;
	solver->app->lua.ref()["vectorFieldResolution"] >> vectorFieldResolution;
	
	//create GL buffer
	glGenBuffers(1, &vectorFieldGLBuffer);
	glBindBuffer(GL_ARRAY_BUFFER_ARB, vectorFieldGLBuffer);
	int vectorFieldVolume = 1;
	for (int i = 0; i < solver->app->dim; ++i) {
		vectorFieldVolume *= vectorFieldResolution;
	}
	vectorFieldVertexCount = 3 * 6 * vectorFieldVolume;
	glBufferData(GL_ARRAY_BUFFER_ARB, sizeof(float) * vectorFieldVertexCount, nullptr, GL_DYNAMIC_DRAW_ARB);
	solver->cl.totalAlloc += sizeof(float) * vectorFieldVertexCount;
	glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	//create CL interop
	vectorFieldVertexBuffer = cl::BufferGL(solver->app->clCommon->context, CL_MEM_READ_WRITE, vectorFieldGLBuffer);
	//create transfer kernel
	updateVectorFieldKernel = cl::Kernel(solver->program, "updateVectorField");
	CLCommon::setArgs(updateVectorFieldKernel, vectorFieldVertexBuffer, solver->stateBuffer, solver->app->vectorFieldScale);
}

VectorField::~VectorField() {
	glDeleteBuffers(1, &vectorFieldGLBuffer);	
}

void VectorField::display() {
	if (!solver->app->showVectorField) return;
	
	//glFlush();
	cl::NDRange global;
	switch (solver->app->dim) {
	case 1:
		global = cl::NDRange(vectorFieldResolution);
		break;
	case 2:
		global = cl::NDRange(vectorFieldResolution, vectorFieldResolution);
		break;
	case 3:
		global = cl::NDRange(vectorFieldResolution, vectorFieldResolution, vectorFieldResolution);
		break;
	}
	updateVectorFieldKernel.setArg(2, (real)solver->app->vectorFieldScale);
	solver->app->clCommon->commands.enqueueNDRangeKernel(updateVectorFieldKernel, solver->offsetNd, global, solver->localSize);
	solver->app->clCommon->commands.finish();

	glDisable(GL_DEPTH_TEST);

	glBindBuffer(GL_ARRAY_BUFFER_ARB, vectorFieldGLBuffer);
	glColor3d(1,1,1);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_LINES, 0, vectorFieldVertexCount);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	
	glEnable(GL_DEPTH_TEST);
}

}
}

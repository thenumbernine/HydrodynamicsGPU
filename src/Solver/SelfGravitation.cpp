#include "HydroGPU/Equation/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/SelfGravitation.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"

namespace HydroGPU {
namespace Solver {

SelfGravitation::SelfGravitation(Solver* solver_) : solver(solver_) {}

void SelfGravitation::initBuffers() {
	int volume = solver->getVolume();
	potentialBuffer = solver->cl.alloc(sizeof(real) * volume, "SelfGravitation::potentialBuffer");
	solidBuffer = solver->cl.alloc(sizeof(char) * volume, "SelfGravitation::solidBuffer");
}

void SelfGravitation::initKernels() {
	cl::Program program = solver->program;
	
	gravityPotentialPoissonRelaxKernel = cl::Kernel(program, "gravityPotentialPoissonRelax");
	CLCommon::setArgs(gravityPotentialPoissonRelaxKernel, potentialBuffer, solver->stateBuffer);
	
	calcGravityDerivKernel = cl::Kernel(program, "calcGravityDeriv");
	calcGravityDerivKernel.setArg(1, solver->stateBuffer);
	calcGravityDerivKernel.setArg(2, potentialBuffer);
}

std::vector<std::string> SelfGravitation::getProgramSources() {
	return {"#include \"SelfGravitation.cl\"\n"};
}

void SelfGravitation::resetState(
	std::vector<real>& stateVec,
	std::vector<real>& potentialVec,
	std::vector<char>& solidVec)
{
	cl::CommandQueue commands = solver->commands;
	cl::NDRange globalSize = solver->globalSize;
	cl::NDRange localSize = solver->localSize;
	cl::NDRange offsetNd = solver->offsetNd;
	
	int volume = solver->getVolume();
	
	//if using gravity then use the density field as an initial guess before poisson relaxiation
	if (solver->app->useGravity) {
		for (size_t i = 0; i < volume; ++i) {
			potentialVec[i] = stateVec[0 + solver->numStates() * i];
		}
	}

	//HACK: if the Lua state has a solid filename then load that and use it for the solid channel ...
	std::string solidFilename;
	if ((solver->app->lua["solidFilename"] >> solidFilename).good()) {
		std::shared_ptr<Image::IImage> image_ = Image::system->read(solidFilename);
		std::shared_ptr<Image::Image> image = std::dynamic_pointer_cast<Image::Image>(image_);
		for (int z = 0; z < solver->app->size.s[2]; ++z) {
			for (int y = 0;  y < solver->app->size.s[1]; ++y) {
				for (int x = 0; x < solver->app->size.s[0]; ++x) {
					int cellIndex = x + solver->app->size.s[0] * (y + solver->app->size.s[1] * z);
					int srcX = x * image->getSize()(0) / solver->app->size.s[0];
					int srcY = y * image->getSize()(1) / solver->app->size.s[1];
					srcY = image->getSize()(1) - 1 - srcY;
					int srcZ = z * image->getPlanes() / solver->app->size.s[2];
					unsigned char solid = (*image)(srcX, srcY, 0, srcZ);
					solidVec[cellIndex] = solid > 127;
				}
			}
		}
	}

	commands.enqueueWriteBuffer(potentialBuffer, CL_TRUE, 0, sizeof(real) * volume, potentialVec.data());
	commands.enqueueWriteBuffer(solidBuffer, CL_TRUE, 0, sizeof(char) * volume, solidVec.data());
	
	if (solver->app->useGravity) {
		//solve for gravitational potential via gauss seidel
		for (int i = 0; i < solver->app->gaussSeidelMaxIter; ++i) {
			potentialBoundary();
			commands.enqueueNDRangeKernel(gravityPotentialPoissonRelaxKernel, offsetNd, globalSize, localSize);
		}
	}

	//add potential energy into total energy
	for (int i = 0; i < volume; ++i) {
		int energyTotalIndex = 1 + solver->app->dim;
		stateVec[energyTotalIndex + solver->numStates() * i] += potentialVec[i];
	}

	commands.enqueueWriteBuffer(solver->stateBuffer, CL_TRUE, 0, sizeof(real) * solver->numStates() * volume, stateVec.data());
	commands.finish();
}

void SelfGravitation::applyPotential(real dt) {
	cl::CommandQueue commands = solver->commands;
	cl::NDRange globalSize = solver->globalSize;
	cl::NDRange localSize = solver->localSize;
	cl::NDRange offsetNd = solver->offsetNd;

	//TODO I had an idea of using the potential buffer to create static source fields even in the absense of self-gravitation
	//  ... but I'm getting weird stuff even when the potential buffer *should* be zero
	// so *either* zero your deriv buffers beforehand (which I'm doing now) *or* find where the fill-deriv kernels are missing their writes
	//if (!solver->app->useGravity) return;
	
	solver->integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		if (solver->app->useGravity) {
			for (int i = 0; i < solver->app->gaussSeidelMaxIter; ++i) {
				potentialBoundary();
				commands.enqueueNDRangeKernel(gravityPotentialPoissonRelaxKernel, offsetNd, globalSize, localSize);
			}	
		}

		calcGravityDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcGravityDerivKernel, offsetNd, globalSize, localSize);
		
		solver->boundary();	
	});
}

void SelfGravitation::potentialBoundary() {
	cl::CommandQueue commands = solver->commands;
	cl::NDRange offset, global, local;
	std::shared_ptr<HydroGPU::Equation::SelfGravitationInterface> gravEqn = std::dynamic_pointer_cast<HydroGPU::Equation::SelfGravitationInterface>(solver->equation);
	for (int i = 0; i < solver->app->dim; ++i) {
		for (int minmax = 0; minmax < 2; ++minmax) {
			int boundaryKernelIndex = gravEqn->gravityGetBoundaryKernelForBoundaryMethod(i, minmax);
			if (boundaryKernelIndex < 0 || boundaryKernelIndex >= solver->boundaryKernels.size()) continue;
			cl::Kernel& kernel = solver->boundaryKernels[boundaryKernelIndex][i][minmax];
			CLCommon::setArgs(kernel, potentialBuffer, 1, 0);
			solver->getBoundaryRanges(i, offset, global, local);
			commands.enqueueNDRangeKernel(kernel, offset, global, local);
		}
	}
}
	
cl::Buffer SelfGravitation::getPotentialBuffer() {
	return potentialBuffer;
}

cl::Buffer SelfGravitation::getSolidBuffer() {
	return solidBuffer;
}

}
}

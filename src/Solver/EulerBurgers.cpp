#include "HydroGPU/Solver/EulerBurgers.h"
#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"

//temporary...
#include "HydroGPU/Integrator/RungeKutta4.h"

namespace HydroGPU {
namespace Solver {

EulerBurgers::EulerBurgers(
	HydroGPUApp* app_)
: Super(app_)
{
}

void EulerBurgers::init() {
	Super::init();
	
	//temporary for implicit
	if (integrator->isImplicit()) {
		pressureIntegrator = std::make_shared<HydroGPU::Integrator::RungeKutta4>(this);
	} else {
		pressureIntegrator = integrator;
	}
}

void EulerBurgers::initBuffers() {
	Super::initBuffers();
	
	int volume = getVolume();

	interfaceVelocityBuffer = clAlloc(sizeof(real) * volume * app->dim, "EulerBurgers::interfaceVelocityBuffer");

	//TODO move this to integrator
	derivStateCoeffBuffer = createDStateDtMatrix();

	//This is just used to add/subtract to create the derivStateCoeffBuffer one function call later.
	//I could get rid of this, but it would mean doubling up the number of calculations in the calcDerivCoeffsFromFluxCoeffs kernel.
	//real fluxStateCoeffBuffer[index:size][side:dim][l/r:2][coeff:numStates]
	fluxStateCoeffBuffer = clAlloc(sizeof(real) * numStates() * 2 * app->dim * volume, "EulerBurgers::fluxStateCoeffBuffer");

	pressureBuffer = clAlloc(sizeof(real) * volume, "EulerBurgers::pressureBuffer");

	//zero interface and flux
	commands.enqueueFillBuffer(interfaceVelocityBuffer, 0.f, 0, sizeof(real) * volume * app->dim);
}

/*
coefficients of the left and right neighboring state buffer to multiply and produce the interface flux
real derivStateCoeffBuffer[index:size][neighbor:2*dim+1][coeff:numStates]
*/
cl::Buffer EulerBurgers::createDStateDtMatrix() {
	return clAlloc(sizeof(real) * numStates() * (1 + 2 * app->dim) * getVolume(), "EulerBurgers::createDStateDtMatrix");
}

void EulerBurgers::applyDStateDtMatrix(cl::Buffer result, cl::Buffer x) {
	calcDerivFromStateCoeffsKernel.setArg(0, result);
	calcDerivFromStateCoeffsKernel.setArg(1, x);
	commands.enqueueNDRangeKernel(calcDerivFromStateCoeffsKernel, offsetNd, globalSize, localSize);
}

void EulerBurgers::initKernels() {
	Super::initKernels();
	
	findMinTimestepKernel = cl::Kernel(program, "findMinTimestep");
	app->setArgs(findMinTimestepKernel, dtBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app->setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer, selfgrav->solidBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app->setArgs(calcFluxKernel, fluxStateCoeffBuffer, stateBuffer, interfaceVelocityBuffer, selfgrav->solidBuffer);
	
	calcDerivCoeffsFromFluxCoeffsKernel = cl::Kernel(program, "calcDerivCoeffsFromFluxCoeffs");
	app->setArgs(calcDerivCoeffsFromFluxCoeffsKernel, derivStateCoeffBuffer, fluxStateCoeffBuffer, selfgrav->solidBuffer);
	
	calcDerivFromStateCoeffsKernel = cl::Kernel(program, "calcDerivFromStateCoeffs");
	calcDerivFromStateCoeffsKernel.setArg(2, derivStateCoeffBuffer);
	calcDerivFromStateCoeffsKernel.setArg(3, selfgrav->solidBuffer);		
	
	computePressureKernel = cl::Kernel(program, "computePressure");
	app->setArgs(computePressureKernel, pressureBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);
	
	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
	diffuseMomentumKernel.setArg(1, pressureBuffer);
	diffuseMomentumKernel.setArg(2, selfgrav->solidBuffer);
	
	diffuseWorkKernel = cl::Kernel(program, "diffuseWork");
	diffuseWorkKernel.setArg(1, stateBuffer);
	diffuseWorkKernel.setArg(2, pressureBuffer);
	diffuseWorkKernel.setArg(3, selfgrav->solidBuffer);
}

void EulerBurgers::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::Euler>(this);
}

std::vector<std::string> EulerBurgers::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"EulerBurgers.cl\"\n");
	return sources;
}

real EulerBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(findMinTimestepKernel, offsetNd, globalSize, localSize);

	return findMinTimestep();
}

void EulerBurgers::step(real dt) {
	for (int side = 0; side < app->dim; ++side) {
		integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
			//both integrators do this ...
			
			calcInterfaceVelocityKernel.setArg(3, side);
			commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize);
			
			calcFluxKernel.setArg(4, dt);
			calcFluxKernel.setArg(5, side);
			commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize);

			calcDerivCoeffsFromFluxCoeffsKernel.setArg(3, side);
			commands.enqueueNDRangeKernel(calcDerivCoeffsFromFluxCoeffsKernel, offsetNd, globalSize, localSize);

			//dirty hack for now, while I restructure ...
			if (!integrator->isImplicit()) {
				//explicit does this:
				//Solver needs to provide a function for combining coefficient buffer and state vector class
				calcDerivFromStateCoeffsKernel.setArg(0, derivBuffer);
				calcDerivFromStateCoeffsKernel.setArg(1, stateBuffer);
				calcDerivFromStateCoeffsKernel.setArg(4, side);
				commands.enqueueNDRangeKernel(calcDerivFromStateCoeffsKernel, offsetNd, globalSize, localSize);
			}	
			/*
			implicit uses derivStateCoeffBuffer, 
			abstracted through applyDStateDtMatrix
			which is called after the callback is finished
			
			Other implicit integrators might use multiple buffers of matrix coefficients
			(like MoL RK4 maybe?),
			and in that case the callback() will need to accept the coeff buffer
			...as an alternative to the derivBuffer?
			...or abstract the coeff + state -> deriv function to the SOlver interface
				and have the ExplicitIntegrator's call that to compress the states...
			*/
		});
	}
	boundary();

	selfgrav->applyPotential(dt);
	
	//the Hydrodynamics ii paper says it's important to diffuse momentum before work
	pressureIntegrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		
		commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize);

		diffuseMomentumKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize);

		//calcDerivFromStateCoeffsKernel.setArg(0, derivBuffer);
		//calcDerivFromStateCoeffsKernel.setArg(4, side);
		//commands.enqueueNDRangeKernel(calcDerivFromStateCoeffsKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	pressureIntegrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		//computePressureFunc(pressureBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);
		diffuseWorkKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

}
}


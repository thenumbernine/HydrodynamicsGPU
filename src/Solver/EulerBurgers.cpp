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
, calcCFLEvent("calcCFL")
, calcInterfaceVelocityEvent("calcInterfaceVelocity")
, calcFluxEvent("calcFlux")
, computePressureEvent("computePressure")
, diffuseMomentumEvent("diffuseMomentum")
, diffuseWorkEvent("diffuseWork")
{
}

void EulerBurgers::init() {
	Super::init();
	
	cl::Context context = app->context;
	
	if (!app->useFixedDT) {
		entries.push_back(&calcCFLEvent);
	}
	entries.push_back(&calcInterfaceVelocityEvent);
	entries.push_back(&calcFluxEvent);
	entries.push_back(&computePressureEvent);
	entries.push_back(&diffuseMomentumEvent);
	entries.push_back(&diffuseWorkEvent);

//temporary for implicit
if (integrator->isImplicit()) {
	pressureIntegrator = std::make_shared<HydroGPU::Integrator::RungeKutta4>(this);
} else {
	pressureIntegrator = integrator;
}
}

void EulerBurgers::initBuffers() {
	Super::initBuffers();
	
	cl::Context context = app->context;
	
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
	app->setArgs(calcDerivFromStateCoeffsKernel, result, x, derivStateCoeffBuffer, selfgrav->solidBuffer);		
	commands.enqueueNDRangeKernel(calcDerivFromStateCoeffsKernel, offsetNd, globalSize, localSize);
}

void EulerBurgers::initKernels() {
	Super::initKernels();
	
	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app->setArgs(calcCFLKernel, cflBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer, app->cfl);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app->setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer, selfgrav->solidBuffer);

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app->setArgs(calcFluxKernel, fluxStateCoeffBuffer, stateBuffer, interfaceVelocityBuffer, selfgrav->solidBuffer, dtBuffer);
	
	calcDerivCoeffsFromFluxCoeffsKernel = cl::Kernel(program, "calcDerivCoeffsFromFluxCoeffs");

	calcDerivFromStateCoeffsKernel = cl::Kernel(program, "calcDerivFromStateCoeffs");

	computePressureKernel = cl::Kernel(program, "computePressure");
	app->setArgs(computePressureKernel, pressureBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);

	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
	//app->setArgs(diffuseMomentumKernel, derivStateCoeffBuffer, pressureBuffer, selfgrav->solidBuffer);
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

void EulerBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, nullptr, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void EulerBurgers::step() {
	
	for (int side = 0; side < app->dim; ++side) {
		integrator->integrate([&](cl::Buffer derivBuffer) {
			//both integrators do this ...
			
			calcInterfaceVelocityKernel.setArg(3, side);
			commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize, nullptr, &calcInterfaceVelocityEvent.clEvent);
		
			calcFluxKernel.setArg(5, side);
			commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, nullptr, &calcFluxEvent.clEvent);
			
			app->setArgs(calcDerivCoeffsFromFluxCoeffsKernel, derivStateCoeffBuffer, fluxStateCoeffBuffer, selfgrav->solidBuffer, side);
			commands.enqueueNDRangeKernel(calcDerivCoeffsFromFluxCoeffsKernel, offsetNd, globalSize, localSize);

			//dirty hack for now, while I restructure ...
			if (!integrator->isImplicit()) {
				//explicit does this:
				//Solver needs to provide a function for combining coefficient buffer and state vector class
			
				app->setArgs(calcDerivFromStateCoeffsKernel, derivBuffer, stateBuffer, derivStateCoeffBuffer, selfgrav->solidBuffer, side);
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

	selfgrav->applyPotential();
	
	//the Hydrodynamics ii paper says it's important to diffuse momentum before work
	pressureIntegrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize, nullptr, &computePressureEvent.clEvent);
		
		diffuseMomentumKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize, nullptr, &diffuseWorkEvent.clEvent);
		
		//commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize, nullptr, &diffuseMomentumEvent.clEvent);	
		//calcDerivFromStateCoeffsKernel.setArg(0, derivBuffer);
		//commands.enqueueNDRangeKernel(calcDerivFromStateCoeffsKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	pressureIntegrator->integrate([&](cl::Buffer derivBuffer) {
		//commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize, nullptr, &computePressureEvent.clEvent);
		diffuseWorkKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize, nullptr, &diffuseWorkEvent.clEvent);
	});
	boundary();
}

}
}


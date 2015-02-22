#pragma once

#include "HydroGPU/Solver/Solver.h"
#include <memory>

namespace HydroGPU {
namespace Solver {

template<typename Parent>
struct SelfGravitationBehavior : public Parent {
	typedef Parent Super;
	using Super::Super;
protected:
	std::shared_ptr<HydroGPU::Solver::SelfGravitation> selfgrav;

public:
	virtual void init() {
		selfgrav = std::make_shared<SelfGravitation>(this);
		Super::init();
	}

protected:
	virtual void initBuffers() {
		Super::initBuffers();
		selfgrav->initBuffers();
	}

	virtual void initKernels() {
		Super::initKernels();
		selfgrav->initKernels();
	}

	virtual std::vector<std::string> getProgramSources() {
		std::vector<std::string> sources = Super::getProgramSources();
		std::vector<std::string> added = selfgrav->getProgramSources();
		sources.insert(sources.end(), added.begin(), added.end());
		return sources;
	}

	//reads the potential energy buffer (in addition to the original state buffer)
	//I'm sure there's a better way I could do this
	struct EulerResetStateContext : public Super::ResetStateContext {
		std::vector<real> potentialVec;
		
		EulerResetStateContext(int volume, int numStates)
		: Super::ResetStateContext(volume, numStates)
		, potentialVec(volume)
		{}
	};
	
	virtual std::shared_ptr<Solver::ResetStateContext> createResetStateContext();
	virtual void resetStateCell(std::shared_ptr<Solver::ResetStateContext> ctx, int index, const std::vector<real>& cellResults);
	virtual void resetStateDone(std::shared_ptr<Solver::ResetStateContext> ctx);
};

template<typename Parent>
std::shared_ptr<Solver::ResetStateContext> SelfGravitationBehavior<Parent>::createResetStateContext() {
	return std::make_shared<EulerResetStateContext>(Super::getVolume(), Super::numStates());
}

template<typename Parent>
void SelfGravitationBehavior<Parent>::resetStateCell(std::shared_ptr<Solver::ResetStateContext> ctx_, int index, const std::vector<real>& cellResults) {
	Super::resetStateCell(ctx_, index, cellResults);	//note that ctx->state will be incremented after this
	std::shared_ptr<EulerResetStateContext> ctx = std::dynamic_pointer_cast<EulerResetStateContext>(ctx_);

	//base class executes Lua init callback function once per pixel ... I only want to cycle through this loop once
	ctx->potentialVec[index] = cellResults[cellResults.size()-1];
}

template<typename Parent>
void SelfGravitationBehavior<Parent>::resetStateDone(std::shared_ptr<Solver::ResetStateContext> ctx_) {
	std::shared_ptr<EulerResetStateContext> ctx = std::dynamic_pointer_cast<EulerResetStateContext>(ctx_);
	selfgrav->resetState(ctx->potentialVec, ctx->stateVec);
}


}
}

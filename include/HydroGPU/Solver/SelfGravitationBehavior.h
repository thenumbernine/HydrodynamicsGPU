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

	struct Converter : public Super::Converter {
		typedef typename Super::Converter Super;
		std::vector<real> potentialVec;
		
		Converter(Solver* solver)
		: Super(solver)
		, potentialVec(solver->getVolume())
		{}
	
		virtual void readCell(int index, const std::vector<real>& cellResults) {
			Super::readCell(index, cellResults);
			potentialVec[index] = cellResults[cellResults.size()-1];
		}
		virtual void toGPU() {
			Super::toGPU();
			dynamic_cast<SelfGravitationBehavior*>(Super::solver)->selfgrav->resetState(potentialVec, Super::stateVec);
		}
	};
	friend struct Converter;

	virtual std::shared_ptr<Solver::Converter> createConverter() {
		return std::make_shared<Converter>(this);
	}
};

}
}

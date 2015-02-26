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
	
		virtual void setValues(int index, const std::vector<real>& cellValues) {
			Super::setValues(index, cellValues);
			potentialVec[index] = cellValues[cellValues.size()-1];
		}
		virtual void toGPU() {
			Super::toGPU();
			SelfGravitationBehavior* owner = dynamic_cast<SelfGravitationBehavior*>(Super::solver);
			owner->selfgrav->resetState(potentialVec, Super::stateVec);
		}
		virtual void fromGPU() {
			Super::fromGPU();
			SelfGravitationBehavior* owner = dynamic_cast<SelfGravitationBehavior*>(Super::solver);
			owner->commands.enqueueReadBuffer(owner->selfgrav->potentialBuffer, CL_TRUE, 0, sizeof(real) * owner->getVolume(), potentialVec.data());
			owner->commands.finish();
		}
		virtual real getValue(int index, int channel) {
			if (channel == Super::solver->numStates()) return potentialVec[index];
			return Super::getValue(index, channel);
		}
	};
	friend struct Converter;

	virtual std::shared_ptr<Solver::Converter> createConverter() {
		return std::make_shared<Converter>(this);
	}

	virtual std::vector<std::string> getSaveChannelNames() {
		std::vector<std::string> channelNames = Super::getSaveChannelNames();
		channelNames.push_back("potential");
		return channelNames;
	}
};

}
}

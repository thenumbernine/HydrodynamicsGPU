#pragma once

#include "HydroGPU/Solver/MHDRemoveDivergence.h"
#include <memory>

namespace HydroGPU {
namespace Solver {

template<typename Parent>
struct MHDRemoveDivergenceBehavior : public Parent {
	typedef Parent Super;
	using Super::Super;
protected:
	std::shared_ptr<MHDRemoveDivergence> divfree;

public:
	virtual void init() {
		divfree = std::make_shared<MHDRemoveDivergence>(this);
		Super::init();
		divfree->init();
	}

protected:
	virtual std::vector<std::string> getProgramSources() {
		std::vector<std::string> sources = Super::getProgramSources();
		std::vector<std::string> added = divfree->getProgramSources();
		sources.insert(sources.end(), added.begin(), added.end());
		return sources;
	}
};

}
}

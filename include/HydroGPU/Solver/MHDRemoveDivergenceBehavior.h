#pragma once

#include "HydroGPU/Solver/MHDRemoveDivergence.h"
#include <memory>

namespace HydroGPU {
namespace Solver {

//pure virtual
struct MHDRemoveDivergenceInterface {
	virtual cl::Buffer getMagneticFieldDivergenceBuffer() = 0;
};

template<typename Super_>
struct MHDRemoveDivergenceBehavior : public Super_, public MHDRemoveDivergenceInterface {
	using Super = Super_;
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

public:
	//MHDRemoveDivergenceInterface 
	virtual cl::Buffer getMagneticFieldDivergenceBuffer() {
		return divfree->getMagneticFieldDivergenceBuffer();
	}
};

}
}

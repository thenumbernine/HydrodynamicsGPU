#pragma once

#include <string>
#include <memory>

namespace HydroGPU {
namespace Equation {
struct Equation;
}
namespace Solver {

struct ISolver {
	virtual void init() = 0;
	virtual void resetState() = 0;
	virtual std::string name() const = 0;
	virtual std::shared_ptr<Equation::Equation> getEquation() const = 0;
};

}
}

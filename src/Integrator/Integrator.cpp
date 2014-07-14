#include "HydroGPU/Integrator/Integrator.h"
#include "HydroGPU/Solver/Solver.h"

namespace HydroGPU {
namespace Integrator {

Integrator::Integrator(HydroGPU::Solver::Solver& solver_)
: solver(solver_)
{}

}
}


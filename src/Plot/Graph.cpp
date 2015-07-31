#include "HydroGPU/Plot/Graph.h"
#include "HydroGPU/Solver/Solver.h"

namespace HydroGPU {
namespace Plot {

Graph::Graph(HydroGPU::Solver::Solver* solver_)
: solver(solver_)
, tex(GLuint())
{}

Graph::~Graph() {
	glDeleteTextures(1, &tex);
}

void Graph::display() {



}

}
}

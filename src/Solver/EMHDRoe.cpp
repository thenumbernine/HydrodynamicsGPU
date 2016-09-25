#include "HydroGPU/Solver/EMHDRoe.h"
#include "HydroGPU/Equation/EMHD.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

EMHDRoe::EMHDRoe(HydroGPUApp* app_)
: euler(app_), maxwell(app_) {}

void EMHDRoe::init() {
	euler.init();
	maxwell.init();

	//typical Solver::init() contents:
	createEquation();
}

void EMHDRoe::createEquation() {
	equation = std::make_shared<Equation::EMHD>(euler.app);
}

void EMHDRoe::resetState() {
	//TODO set aside an initState for both EM and Maxwell ... 
	// ... or add electric field to the default initState variables
	euler.resetState();
	maxwell.resetState();
}

void EMHDRoe::update() {
	//TODO keep track of time and update whoever is furthest behind
	euler.update();
	maxwell.update();
}

}
}

#include "HydroGPU/Plot/Plot2D.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include <OpenGL/gl.h>

namespace HydroGPU {
namespace Plot {
	
Plot2D::Plot2D(HydroGPU::Solver::Solver& solver)
: Super(solver)
{
}

void Plot2D::display() {
	glLoadIdentity();
	glTranslatef(-viewPos(0), -viewPos(1), 0);
	glScalef(viewZoom, viewZoom, viewZoom);
	
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0,0); glVertex2f(solver.app.xmin.s[0], solver.app.xmin.s[1]);
	glTexCoord2f(1,0); glVertex2f(solver.app.xmax.s[0], solver.app.xmin.s[1]);
	glTexCoord2f(1,1); glVertex2f(solver.app.xmax.s[0], solver.app.xmax.s[1]);
	glTexCoord2f(0,1); glVertex2f(solver.app.xmin.s[0], solver.app.xmax.s[1]);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
}

}
}


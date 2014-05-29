DIST_FILENAME=hydro-gpu
DIST_TYPE=app

include ../GLApp/Makefile.mk

MACROS+=__CL_ENABLE_EXCEPTIONS
MACROS+=__CL_OVERRIDE_ERROR_STRINGS
INCLUDE+=../GLApp/include
INCLUDE+=../Profiler/include
INCLUDE+=../TensorMath/include
INCLUDE+=res/include
LDFLAGS+= -lGLApp
LDFLAGS+= -framework OpenCL
LDFLAGS+= -L../Profiler/dist/$(PLATFORM)/$(BUILD) -lProfiler

$(DIST):: $(OBJPATHS)
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib ../GLApp/dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib $@
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libProfiler.dylib ../Profiler/dist/$(PLATFORM)/$(BUILD)/libProfiler.dylib $@


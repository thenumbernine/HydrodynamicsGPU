DIST_FILENAME=hydro-gpu
DIST_TYPE=app

include ../GLApp/Makefile.mk

CFLAGS_BASE+=-D__CL_ENABLE_EXCEPTIONS
CFLAGS_BASE+=-D__CL_OVERRIDE_ERROR_STRINGS
INCLUDE_BASE+=../GLApp/include
INCLUDE_BASE+=../Profiler/include
INCLUDE_BASE+=../TensorMath/include
INCLUDE_BASE+=res/include
LDFLAGS_BASE+= -lGLApp
LDFLAGS_BASE+= -framework OpenCL
LDFLAGS_BASE+= -L../Profiler/dist/$(PLATFORM)/$(BUILD) -lProfiler

$(DIST):: $(OBJPATHS)
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib ../GLApp/dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib $@
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libProfiler.dylib ../Profiler/dist/$(PLATFORM)/$(BUILD)/libProfiler.dylib $@


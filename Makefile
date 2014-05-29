DIST_FILENAME=hydro-gpu
DIST_TYPE=app

include ../GLApp/Makefile.mk

CFLAGS_BASE+= -I../GLApp/include
CFLAGS_BASE+= -I../Profiler/include
CFLAGS_BASE+= -I../TensorMath/include
CFLAGS_BASE+= -Ires/include
LDFLAGS_BASE+= -lGLApp
LDFLAGS_BASE+= -framework OpenCL
LDFLAGS_BASE+= -L../Profiler/dist/$(PLATFORM)/$(BUILD) -lProfiler

$(DIST):: $(OBJPATHS)
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib ../GLApp/dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib $@
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libProfiler.dylib ../Profiler/dist/$(PLATFORM)/$(BUILD)/libProfiler.dylib $@


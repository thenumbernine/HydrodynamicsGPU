DIST_FILENAME=hydro-gpu
DIST_TYPE=app

include ../Common/Base.mk
include ../Common/Include.mk
include ../GLApp/Include.mk
include ../CLApp/Include.mk
include ../TensorMath/Include.mk
include ../Profiler/Include.mk

INCLUDE+=res/include

$(DIST):: $(OBJPATHS)
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libCommon.dylib ../Common/dist/$(PLATFORM)/$(BUILD)/libCommon.dylib $@
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib ../GLApp/dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib $@
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libProfiler.dylib ../Profiler/dist/$(PLATFORM)/$(BUILD)/libProfiler.dylib $@
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libCLApp.dylib ../CLApp/dist/$(PLATFORM)/$(BUILD)/libCLApp.dylib $@


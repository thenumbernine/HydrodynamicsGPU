DIST_FILENAME=hydro-gpu
DIST_TYPE=app

include ../GLApp/Makefile.mk

CFLAGS_BASE+= -I../GLApp/include -I../TensorMath/include -Ires/include
LDFLAGS_BASE+= -lGLApp -lpng -ltiff -ljpeg -lcfitsio -framework OpenCL

$(DIST):: $(OBJPATHS)
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib ../GLApp/dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib $@


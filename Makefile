# PLATFORM: osx
# BUILD: debug, release

DIST_FILENAME=hydro-gpu
DIST_TYPE=app

OBJECTS=main.o
SOURCES=$(shell find src -type f)
HEADERS=$(shell find include -type f)

include ../GLApp/Makefile.mk

CFLAGS_BASE+= -I../GLApp/include -I../Image/include -I../TensorMath/include -Ires/include
LDFLAGS_BASE+= -L../Image/dist/$(PLATFORM)/$(BUILD) -limage -lpng -ltiff -ljpeg -lcfitsio -framework OpenCL

$(DIST):: $(OBJPATHS)
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib ../GLApp/dist/$(PLATFORM)/$(BUILD)/libGLApp.dylib $@
	install_name_tool -change dist/$(PLATFORM)/$(BUILD)/libimage.dylib ../Image/dist/$(PLATFORM)/$(BUILD)/libimage.dylib $@


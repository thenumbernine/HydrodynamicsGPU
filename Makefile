HYDROGPU_PATH=$(dir $(lastword $(MAKEFILE_LIST)))
DIST_FILENAME=HydroGPU
DIST_TYPE=app

include ../Common/Base.mk
include ../Common/Include.mk
include ../GLApp/Include.mk
include ../CLApp/Include.mk
include ../Tensor/Include.mk
include ../Profiler/Include.mk
include ../Image/Include.mk
include ../Shader/Include.mk
include ../LuaCxx/Include.mk

LUA_EXT_PATH=$(HYDROGPU_PATH)/../lua/ext
LUA_SYMMATH_PATH=$(HYDROGPU_PATH)/../lua/symmath
DIST_RESOURCE_PATH=$(DISTDIR)/$(call concat,$(call buildVar,DIST_PREFIX)$(DIST_FILENAME)).app/Contents/Resources/

INCLUDE+=res/include
post_builddist_osx_app::
	@echo "copying Lua scripts..."
	rsync -avm --include='*.lua' -f 'hide,! */' $(LUA_EXT_PATH) $(DIST_RESOURCE_PATH)
	rsync -avm --include='*.lua' -f 'hide,! */' $(LUA_SYMMATH_PATH) $(DIST_RESOURCE_PATH)
	@echo "done copying Lua scripts."

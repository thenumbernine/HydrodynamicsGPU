HYDROGPU_PATH=$(dir $(lastword $(MAKEFILE_LIST)))
DIST_FILENAME=HydroGPU
DIST_TYPE=app

include ../Common/Base.mk
include ../Common/Include.mk
include ../GLApp/Include.mk
include ../CLCommon/Include.mk
include ../Tensor/Include.mk
include ../Profiler/Include.mk
include ../Image/Include.mk
include ../Shader/Include.mk
include ../LuaCxx/Include.mk

MACROS+= AMD_SUCKS
DYNAMIC_LIBS+= libnanogui.dylib
LUA_EXT_PATH=$(HYDROGPU_PATH)/../lua/ext
LUA_SYMMATH_PATH=$(HYDROGPU_PATH)/../lua/symmath
DIST_RESOURCE_PATH=$(DISTDIR)/$(call concat,$(call buildVar,DIST_PREFIX)$(DIST_FILENAME)).app/Contents/Resources/

# $(call copyTreeOfType, pattern, path from, path to)
copyTreeOfType = rsync -avm --include='$1' -f 'hide,! */' $2 $3

INCLUDE+=res/include
post_builddist_osx_app::
	@echo "copying Lua scripts..."
	$(call copyTreeOfType,*.lua,$(LUA_EXT_PATH),$(DIST_RESOURCE_PATH))
	$(call copyTreeOfType,*.lua,$(LUA_SYMMATH_PATH),$(DIST_RESOURCE_PATH))
	@echo "done copying Lua scripts."

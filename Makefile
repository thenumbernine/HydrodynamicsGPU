HYDROGPU_PATH=$(dir $(lastword $MAKEFILE_LIST))
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
include ../ImGuiCommon/Include.mk

#override the original -std=c++11 that I have baked in my Base.mk
CPPVER= c++14

LUA_PATH=$(HOME)/Projects/lua
LUA_EXT_PATH=$(LUA_PATH)/ext
LUA_SYMMATH_PATH=$(LUA_PATH)/symmath
DIST_RESOURCE_PATH=$(dir $(DIST))/../Resources/

# $(call copyTreeOfType, pattern, path from, path to)
copyTreeOfType = rsync -avm --include='$1' -f 'hide,! */' $2 $3

INCLUDE+=res/include
post_builddist_osx_app::
	@echo "copying Lua scripts..."
	$(call copyTreeOfType,*.lua,$(LUA_EXT_PATH),$(DIST_RESOURCE_PATH))
	$(call copyTreeOfType,*.lua,$(LUA_SYMMATH_PATH),$(DIST_RESOURCE_PATH))
	@echo "done copying Lua scripts."

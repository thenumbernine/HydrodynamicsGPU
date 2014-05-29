#include "HydroGPU/Solver.h"
#include "HydroGPU/RoeSolver.h"
#include "GLApp/GLApp.h" 
#include "Profiler/Profiler.h"
#include "TensorMath/Vector.h"
#include "Common/Exception.h"
#include "Common/Macros.h"
#include <SDL2/SDL.h>
#include <OpenCL/opencl.h>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

struct HydroGPUApp : public GLApp {
	bool useGPU;	//whether we want to request the GPU or CPU.  required prior to init()
	
	GLuint fluidTex;
	GLuint gradientTex;
	
	cl_device_id deviceID;
	cl_context context;
	
	cl_command_queue commands;
	cl_mem fluidTexMem;		//data is written to this buffer before rendering
	cl_mem gradientTexMem;	//as it is written, data is read from this for mapping values to colors

	Solver *solver;

	Vector<size_t,DIM> local_size;
	Vector<size_t,DIM> global_size;
	cl_int2 size;
	  
	bool leftButtonDown;
	bool leftShiftDown;
	bool rightShiftDown;
	int doUpdate;	//0 = no, 1 = continuous, 2 = single step
	float viewZoom;
	float viewPosX;
	float viewPosY;

	HydroGPUApp();

	cl_platform_id getPlatformID();
	cl_device_id getDeviceID(cl_platform_id platformID);

	virtual int main(int argc, char **argv);
	virtual void init();
	virtual void shutdown();
	virtual void resize(int width, int height);
	virtual void update();
	virtual void sdlEvent(SDL_Event &event);
};

HydroGPUApp::HydroGPUApp()
: GLApp()
, useGPU(true)
, fluidTex(GLuint())
, gradientTex(GLuint())
, deviceID(cl_device_id())
, context(cl_context())
, commands(cl_command_queue())
, fluidTexMem(cl_mem())
, gradientTexMem(cl_mem())
, leftButtonDown(false)
, leftShiftDown(false)
, rightShiftDown(false)
, doUpdate(1)
, viewZoom(1.f)
, viewPosX(0.f)
, viewPosY(0.f)
{
}

int HydroGPUApp::main(int argc, char **argv) {
	for (int i = 0; i < DIM; ++i) {
		size.s[i] = 256;
	}

	for (int i = 0; i < argc; ++i) {
		if (!strcmp(argv[i], "--cpu")) {
			useGPU = false;
		}
		if (i < argc-2) {
			if (!strcmp(argv[i], "--size")) {
				size.s[0] = atoi(argv[++i]);
				size.s[1] = atoi(argv[++i]);
			}
		}
	}
	return GLApp::main(argc, argv);
}

cl_platform_id HydroGPUApp::getPlatformID() {
	cl_uint numPlatforms = 0;
	int err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms == 0) throw Exception() << "failed to query number of CL platforms.  got error " << err;
	
	std::vector<cl_platform_id> platformIDs(numPlatforms);
	err = clGetPlatformIDs(numPlatforms, &platformIDs[0], NULL);
	if (err != CL_SUCCESS) throw Exception() << "failed to query CL platforms.  got error " << err;
 
	std::for_each(platformIDs.begin(), platformIDs.end(), [&](cl_platform_id platformID) {
		std::cout << "platform " << platformID << std::endl;
		std::pair<cl_uint, const char *> queries[] = {
			std::pair<cl_uint, const char *>(CL_PLATFORM_NAME, "name"),
			std::pair<cl_uint, const char *>(CL_PLATFORM_VENDOR, "vendor"),
			std::pair<cl_uint, const char *>(CL_PLATFORM_VERSION, "version"),
			std::pair<cl_uint, const char *>(CL_PLATFORM_PROFILE, "profile"),
			std::pair<cl_uint, const char *>(CL_PLATFORM_EXTENSIONS, "extensions"),
		};
		std::for_each(queries, queries + numberof(queries), [&](std::pair<cl_uint, const char *> query) {	
			size_t param_value_size_ret = 0;
			err = clGetPlatformInfo(platformID, query.first, 0, NULL, &param_value_size_ret);
			if (err != CL_SUCCESS) throw Exception() << "clGetPlatformInfo failed to query " << query.second << " (" << query.first << ") for platform " << platformID << " with error " << err;		
		
			std::string param_value(param_value_size_ret, '\0');
			err = clGetPlatformInfo(platformID, query.first, param_value_size_ret, (void*)param_value.c_str(), NULL);
			if (err != CL_SUCCESS) throw Exception() << "clGetPlatformInfo failed for platform " << platformID << " with error " << err;
			
			std::cout << query.second << ":\t" << param_value << std::endl;
		});
		std::cout << std::endl;
	});

	return platformIDs[0];
}

struct DeviceParameterQuery {
	cl_uint param;
	const char *name;
	bool failed;
	DeviceParameterQuery(cl_uint param_, const char *name_) : param(param_), name(name_), failed(false) {}
	virtual void query(cl_device_id deviceID) = 0;
	std::string tostring() {
		if (failed) return "-failed-";
		return toStringType();
	}
	virtual std::string toStringType() = 0;
};

template<typename Type>
struct DeviceParameterQueryType : public DeviceParameterQuery {
	Type value;
	DeviceParameterQueryType(cl_uint param_, const char *name_) : DeviceParameterQuery(param_, name_), value(Type()) {}
	virtual void query(cl_device_id deviceID) {
		int err = clGetDeviceInfo(deviceID, param, sizeof(value), &value, NULL);
		if (err != CL_SUCCESS) {
			//throw Exception() << "clGetDeviceInfo failed to query " << name << " (" << param << ") for device " << deviceID << " with error " << err;
			failed = true;
			return;
		}
	}
	virtual std::string toStringType() {
		std::stringstream ss;
		ss << value;
		return ss.str();
	}
};

template<>
struct DeviceParameterQueryType<char*> : public DeviceParameterQuery {
	std::string value;
	DeviceParameterQueryType(cl_uint param_, const char *name_) : DeviceParameterQuery(param_, name_) {}
	virtual void query(cl_device_id deviceID) {
		size_t size = 0;
		int err = clGetDeviceInfo(deviceID, param, 0, NULL, &size);
		if (err != CL_SUCCESS) {
			//throw Exception() << "clGetDeviceInfo failed to query " << name << " (" << param << ") for device " << deviceID << " with error " << err;
			failed = true;
			return;
		}
	
		value = std::string(size, '\0');
		err = clGetDeviceInfo(deviceID, param, size, (void*)value.c_str(), NULL);
		if (err != CL_SUCCESS) {
			//throw Exception() << "clGetDeviceInfo failed to query " << name << " (" << param << ") for device " << deviceID << " with error " << err;
			failed = true;
			return;
		}
	}
	virtual std::string toStringType() { return value; }
};

//has to be a class separate of DeviceParameterQueryType because some types are used with both (cl_device_fp_config is typedef'd as a cl_uint)
template<typename Type>
struct DeviceParameterQueryEnumType : public DeviceParameterQuery {
	Type value;
	DeviceParameterQueryEnumType(cl_uint param_, const char *name_) : DeviceParameterQuery(param_, name_), value(Type()) {}
	virtual void query(cl_device_id deviceID) {
		int err = clGetDeviceInfo(deviceID, param, sizeof(value), &value, NULL);
		if (err != CL_SUCCESS) {
			//throw Exception() << "clGetDeviceInfo failed to query " << name << " (" << param << ") for device " << deviceID << " with error " << err;
			failed = true;
			return;
		}
	}
	virtual std::vector<std::pair<Type, const char *>> getFlags() = 0;
	virtual std::string toStringType() {
		std::vector<std::pair<Type, const char *>>flags = getFlags();
		Type copy = value;
		std::stringstream ss;
		std::for_each(flags.begin(), flags.end(), [&](std::pair<Type, const char *> flag) {
			if (copy & flag.first) {
				copy -= flag.first;
				ss << "\n\t" << flag.second;
			}
		});
		if (copy) {
			ss << "\n\textra flags: " << copy;
		}
		return ss.str();
	};
};

struct DeviceParameterQueryEnumType_cl_device_fp_config : public DeviceParameterQueryEnumType<cl_device_fp_config> {
	using DeviceParameterQueryEnumType::DeviceParameterQueryEnumType;
	virtual std::vector<std::pair<cl_device_fp_config, const char *>> getFlags() { 
		return std::vector<std::pair<cl_device_fp_config, const char *>>{
			std::pair<cl_device_fp_config, const char *>(CL_FP_DENORM, "CL_FP_DENORM - denorms are supported"),
			std::pair<cl_device_fp_config, const char *>(CL_FP_INF_NAN, "CL_FP_INF_NAN - INF and NaNs are supported"),
			std::pair<cl_device_fp_config, const char *>(CL_FP_ROUND_TO_NEAREST, "CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported"),
			std::pair<cl_device_fp_config, const char *>(CL_FP_ROUND_TO_ZERO, "CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported"),
			std::pair<cl_device_fp_config, const char *>(CL_FP_ROUND_TO_INF, "CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported"),
			std::pair<cl_device_fp_config, const char *>(CL_FP_FMA, "CL_FP_FMA - IEEE754-20080 fused multiply-add is supported"),
			std::pair<cl_device_fp_config, const char *>(CL_FP_SOFT_FLOAT, "CL_FP_SOFT_FLOAT"),
			std::pair<cl_device_fp_config, const char *>(CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT"),
		};
	}
};

struct DeviceParameterQueryEnumType_cl_device_exec_capabilities : public DeviceParameterQueryEnumType<cl_device_exec_capabilities> {
	using DeviceParameterQueryEnumType::DeviceParameterQueryEnumType;
	virtual std::vector<std::pair<cl_device_exec_capabilities, const char *>> getFlags() { 
		return std::vector<std::pair<cl_device_exec_capabilities, const char *>>{
			std::pair<cl_device_exec_capabilities, const char *>(CL_EXEC_KERNEL, "CL_EXEC_KERNEL - The OpenCL device can execute OpenCL kernels"),
			std::pair<cl_device_exec_capabilities, const char *>(CL_EXEC_NATIVE_KERNEL, "CL_EXEC_NATIVE_KERNEL - The OpenCL device can execute native kernels"),
		};
	}
};

cl_device_id HydroGPUApp::getDeviceID(cl_platform_id platformID) {	
	cl_uint numDevices = 0;
	int err = clGetDeviceIDs(platformID, useGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
	if (err != CL_SUCCESS || numDevices == 0) throw Exception() << "failed to query number of CL devices.  got error " << err;

	std::vector<cl_device_id> deviceIDs(numDevices);

	err = clGetDeviceIDs(platformID, useGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, numDevices, &deviceIDs[0], NULL);
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to create a device group!";
	
	std::vector<std::shared_ptr<DeviceParameterQuery>> deviceParameters = {
		std::make_shared<DeviceParameterQueryType<char*>>(CL_DEVICE_NAME, "name"),
		std::make_shared<DeviceParameterQueryType<char*>>(CL_DEVICE_VENDOR, "vendor"),
		std::make_shared<DeviceParameterQueryType<char*>>(CL_DEVICE_VERSION, "version"),
		std::make_shared<DeviceParameterQueryType<char*>>(CL_DRIVER_VERSION, "driver version"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_VENDOR_ID, "vendor id"),
		std::make_shared<DeviceParameterQueryType<cl_platform_id>>(CL_DEVICE_PLATFORM, "platform id"),
		std::make_shared<DeviceParameterQueryType<cl_bool>>(CL_DEVICE_AVAILABLE, "available?"),
		std::make_shared<DeviceParameterQueryType<cl_bool>>(CL_DEVICE_COMPILER_AVAILABLE, "compiler available?"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MAX_CLOCK_FREQUENCY, "max clock freq"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MAX_COMPUTE_UNITS, "cores"),
		std::make_shared<DeviceParameterQueryType<cl_device_type>>(CL_DEVICE_TYPE, "type"),
		std::make_shared<DeviceParameterQueryEnumType_cl_device_fp_config>(CL_DEVICE_ADDRESS_BITS, "fp caps"),	//bitflags: 
		std::make_shared<DeviceParameterQueryEnumType_cl_device_fp_config>(CL_DEVICE_HALF_FP_CONFIG, "half fp caps"),
		std::make_shared<DeviceParameterQueryEnumType_cl_device_fp_config>(CL_DEVICE_SINGLE_FP_CONFIG, "single fp caps"),
		std::make_shared<DeviceParameterQueryType<cl_bool>>(CL_DEVICE_ENDIAN_LITTLE, "little endian?"),
		std::make_shared<DeviceParameterQueryEnumType_cl_device_exec_capabilities>(CL_DEVICE_EXECUTION_CAPABILITIES, "exec caps"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_ADDRESS_BITS, "address space size"),
		std::make_shared<DeviceParameterQueryType<cl_ulong>>(CL_DEVICE_GLOBAL_MEM_SIZE, "global mem size"),
		std::make_shared<DeviceParameterQueryType<cl_ulong>>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "global mem cache size"),
		std::make_shared<DeviceParameterQueryType<cl_device_mem_cache_type>>(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, "global mem cache type"),	//CL_NONE, CL_READ_ONLY_CACHE, and CL_READ_WRITE_CACHE.
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "global mem cache line size"),
		std::make_shared<DeviceParameterQueryType<cl_ulong>>(CL_DEVICE_LOCAL_MEM_SIZE, "local mem size"),
		std::make_shared<DeviceParameterQueryType<cl_device_local_mem_type>>(CL_DEVICE_LOCAL_MEM_TYPE, "local mem type"),	//SRAM, or CL_GLOBAL
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MEM_BASE_ADDR_ALIGN, "mem align"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "data type align"),
		std::make_shared<DeviceParameterQueryType<cl_bool>>(CL_DEVICE_IMAGE_SUPPORT, "image support?"),
		std::make_shared<DeviceParameterQueryType<size_t>>(CL_DEVICE_IMAGE2D_MAX_WIDTH, "image2d max width"),
		std::make_shared<DeviceParameterQueryType<size_t>>(CL_DEVICE_IMAGE2D_MAX_HEIGHT, "image2d max height"),
		std::make_shared<DeviceParameterQueryType<size_t>>(CL_DEVICE_IMAGE3D_MAX_WIDTH, "image3d max width"),
		std::make_shared<DeviceParameterQueryType<size_t>>(CL_DEVICE_IMAGE3D_MAX_HEIGHT, "image3d max height"),
		std::make_shared<DeviceParameterQueryType<size_t>>(CL_DEVICE_IMAGE3D_MAX_DEPTH, "image2d max depth"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MAX_CONSTANT_ARGS, "max __constant args"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "max __constant buffer size"),
		std::make_shared<DeviceParameterQueryType<cl_ulong>>(CL_DEVICE_MAX_MEM_ALLOC_SIZE, "max mem alloc size"),
		std::make_shared<DeviceParameterQueryType<size_t>>(CL_DEVICE_MAX_PARAMETER_SIZE, "max param size"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MAX_READ_IMAGE_ARGS, "max read image objs"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "max write image objs"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MAX_SAMPLERS, "max samplers"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "preferred char vector width"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "preferred short vector width"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "preferred int vector width"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "preferred long vector width"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "preferred float vector width"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "preferred double vector width"),
		std::make_shared<DeviceParameterQueryType<size_t>>(CL_DEVICE_MAX_WORK_GROUP_SIZE, "max items in work-group"),
		std::make_shared<DeviceParameterQueryType<cl_uint>>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "max work item dim"),
		std::make_shared<DeviceParameterQueryType<Vector<size_t,3>>>(CL_DEVICE_MAX_WORK_ITEM_SIZES, "max work item sizes"),
		std::make_shared<DeviceParameterQueryType<char*>>(CL_DEVICE_PROFILE, "profile"),
		std::make_shared<DeviceParameterQueryType<size_t>>(CL_DEVICE_PROFILING_TIMER_RESOLUTION, "profile timer resolution"),
		std::make_shared<DeviceParameterQueryType<cl_command_queue_properties>>(CL_DEVICE_QUEUE_PROPERTIES, "command-queue properties"),
		//std::make_shared<DeviceParameterQueryType<char*>>(CL_DEVICE_EXTENSIONS, "extensions"),
	};

	std::vector<cl_device_id>::iterator deviceIter =
		std::find_if(deviceIDs.begin(), deviceIDs.end(), [&](cl_device_id deviceID)
	{
		std::cout << "device " << deviceID << std::endl;
		std::for_each(deviceParameters.begin(), deviceParameters.end(), [&](std::shared_ptr<DeviceParameterQuery> &query) {
			query->query(deviceID);
			std::cout << query->name << ":\t" << query->tostring() << std::endl;
		});

		size_t param_value_size_ret = 0;
		err = clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, 0, NULL, &param_value_size_ret);
		if (err != CL_SUCCESS) throw Exception() << "clGetDeviceInfo failed for device " << deviceID << " with error " << err;		
	
		std::string param_value(param_value_size_ret, '\0');
		err = clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, param_value_size_ret, (void*)param_value.c_str(), NULL);
		if (err != CL_SUCCESS) throw Exception() << "clGetDeivceInfo failed for device " << deviceID << " with error " << err;

		std::vector<std::string> extensions;
		std::istringstream iss(param_value);
		std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), std::back_inserter<std::vector<std::string>>(extensions));

		std::cout << "extensions:" << std::endl;
		std::for_each(extensions.begin(), extensions.end(), [&](const std::string &s) {
			std::cout << "\t" << s << std::endl;
		});
		std::cout << std::endl;

		std::vector<std::string>::iterator extension = 
			std::find_if(extensions.begin(), extensions.end(), [&](const std::string &s)
		{
			return s == std::string("cl_khr_gl_sharing") 	//i don't have
				|| s == std::string("cl_APPLE_gl_sharing");	//i do have!
		});
	 
		return extension != extensions.end();
	});
	if (deviceIter == deviceIDs.end()) throw Exception() << "failed to find a device capable of GL sharing";
	return *deviceIter;
}
	
void HydroGPUApp::init() {
	GLApp::init();

	int err;
	  
	real noise = real(.01);
	real2 xmin, xmax;
	for (int n = 0; n < DIM; ++n) {
		xmin.s[n] = -.5;
		xmax.s[n] = .5;
	}
	
	std::vector<Cell> cells(size.s[0] * size.s[1]);
	{
		int index[DIM];
		
		Cell *cell = &cells[0];
		//for (index[2] = 0; index[2] < size.s[2]; ++index[2]) {
			for (index[1] = 0; index[1] < size.s[1]; ++index[1]) {
				for (index[0] = 0; index[0] < size.s[0]; ++index[0], ++cell) {
					bool lhs = true;
					for (int n = 0; n < DIM; ++n) {
						cell->x.s[n] = real(xmax.s[n] - xmin.s[n]) * real(index[n]) / real(size.s[n]) + real(xmin.s[n]);
						if (cell->x.s[n] > real(.3) * real(xmax.s[n]) + real(.7) * real(xmin.s[n])) {
							lhs = false;
						}
					}

					for (int m = 0; m < DIM; ++m) {
						cell->interfaces[m].solid = false;
						for (int n = 0; n < DIM; ++n) {
							cell->interfaces[m].x.s[n] = cell->x.s[n];
							if (m == n) {
								cell->interfaces[m].x.s[n] -= real(xmax.s[n] - xmin.s[n]) * real(.5) / real(size.s[n]);
							}
						}
					}

					//sod init
					real density = lhs ? 1. : .1;
					real velocity[DIM];
					real energyKinetic = real();
					for (int n = 0; n < DIM; ++n) {
						velocity[n] = crand() * noise;
						energyKinetic += velocity[n] * velocity[n];
					}
					energyKinetic *= real(.5);
					real energyThermal = 1.;
					real energyTotal = energyKinetic + energyThermal;

					cell->q.s[0] = density;
					for (int n = 0; n < DIM; ++n) {
						cell->q.s[n+1] = density * velocity[n];
					}
					cell->q.s[DIM+1] = density * energyTotal;
				}
			}
		//}
	}

	cl_platform_id platformID = getPlatformID();
	deviceID = getDeviceID(platformID);

	size_t maxWorkGroupSize = 0;
	err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
	if (err != CL_SUCCESS) throw Exception() << "clGetDeviceInfo failed to query CL_DEVICE_MAX_WORK_GROUP_SIZE with error " << err;

	Vector<size_t,3> maxWorkItemSizes;
	err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), &maxWorkItemSizes, NULL);
	if (err != CL_SUCCESS) throw Exception() << "clGetDeviceInfo failed to query CL_DEVICE_MAX_WORK_ITEM_SIZES with error " << err;

	for (int n = 0; n < DIM; ++n) {
		global_size(n) = size.s[n];
		local_size(n) = std::min<size_t>(maxWorkItemSizes(n), size.s[n]);
	}
	while (local_size.volume() > maxWorkGroupSize) {
		for (int n = 0; n < DIM; ++n) {
			local_size(n) = (size_t)ceil((double)local_size(n) * .5);
		}
	}
	//hmm...
	if (!useGPU) local_size(0) >>= 1;
	std::cout << "global_size\t" << global_size << std::endl;
	std::cout << "local_size\t" << local_size << std::endl;

#if PLATFORM_osx
	CGLContextObj kCGLContext = CGLGetCurrentContext();	// GL Context
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext); // Share Group
	cl_context_properties properties[] = {
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
		CL_CONTEXT_PLATFORM, (cl_context_properties)platformID,
		0
	};
#endif
#if PLATFORM_windows
	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext(), // HGLRC handle
		CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC(), // HDC handle
		CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
		0
	};	
#endif

	context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &err);
	if (!context) throw Exception() << "Error: Failed to create a compute context!";
 
	commands = clCreateCommandQueue(context, deviceID, 0, &err);
	if (!commands) throw Exception() << "Error: Failed to create a command queue!";

	//get a texture going for visualizing the output
	glGenTextures(1, &fluidTex);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, size.s[0], size.s[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	if ((err = glGetError()) != 0) throw Exception() << "failed to create GL texture.  got error " << err;
	
	fluidTexMem = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, fluidTex, &err);
	if (!fluidTexMem) throw Exception() << "failed to create CL memory from GL texture.  got error " << err;

	//gradient texture
	{
		glGenTextures(1, &gradientTex);
		glBindTexture(GL_TEXTURE_2D, gradientTex);

		float colors[][3] = {
			{0, 0, 0},
			{0, 0, .5},
			{1, .5, 0},
			{1, 0, 0}
		};

		const int width = 256;
		unsigned char data[width*3];
		for (int i = 0; i < width; ++i) {
			float f = (float)i / (float)width * (float)numberof(colors);
			int ci = (int)f;
			int ci2 = (ci + 1) % numberof(colors);
			float s = f - (float)ci;
			if (ci >= numberof(colors)) {
				ci = numberof(colors)-1;
				s = 0;
			}

			for (int j = 0; j < 3; ++j) {
				data[3 * i + j] = (unsigned char)(255. * (colors[ci][j] * (1.f - s) + colors[ci2][j] * s));
			}
		}

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	gradientTexMem = clCreateFromGLTexture(context, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, gradientTex, &err);
	if (!gradientTexMem) throw Exception() << "failed to create CL memory from GL texture.  got error " << err;

	solver = new RoeSolver(
		deviceID, 
		context, 
		size, 
		commands, 
		cells, 
		xmin, 
		xmax, 
		fluidTexMem, 
		gradientTexMem, 
		local_size.v,
		useGPU);

	std::cout << "Success!" << std::endl;
}

void HydroGPUApp::shutdown() {
	delete solver; solver = NULL;
	glDeleteTextures(1, &fluidTex);
	glDeleteTextures(1, &gradientTex);
	clReleaseContext(context);
	clReleaseCommandQueue(commands);
	clReleaseMemObject(fluidTexMem);
	clReleaseMemObject(gradientTexMem);

	PROFILE_DONE()
}

void HydroGPUApp::resize(int width, int height) {
	GLApp::resize(width, height);	//viewport
	float aspectRatio = (float)width / (float)height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-aspectRatio, aspectRatio, -1., 1., -1., 1.);
	glMatrixMode(GL_MODELVIEW);
}

void HydroGPUApp::update() {
PROFILE_BEGIN_FRAME()
{
PROFILE()
	GLApp::update();	//glclear 
	
	//CPU need to bind beforehand for roe/cpu to use it
	//GPU needs it unbound until after the update
	if (!useGPU) {
		glBindTexture(GL_TEXTURE_2D, fluidTex);
	}

	if (doUpdate) {
		solver->update(commands, fluidTexMem, global_size.v, local_size.v);
		if (doUpdate == 2) doUpdate = 0;
	}

	glPushMatrix();
	glTranslatef(-viewPosX, -viewPosY, 0);
	glScalef(viewZoom, viewZoom, viewZoom);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0,0); glVertex2f(-.5f,-.5f);
	glTexCoord2f(1,0); glVertex2f(.5f,-.5f);
	glTexCoord2f(1,1); glVertex2f(.5f,.5f);
	glTexCoord2f(0,1); glVertex2f(-.5f,.5f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glPopMatrix();

	int err = glGetError();
	if (err) std::cout << "error " << err << std::endl;
}
PROFILE_END_FRAME();
}

void HydroGPUApp::sdlEvent(SDL_Event &event) {
	bool shiftDown = leftShiftDown | rightShiftDown;

	switch (event.type) {
	case SDL_MOUSEMOTION:
		{
			int dx = event.motion.xrel;
			int dy = event.motion.yrel;
			if (leftButtonDown) {
				if (shiftDown) {
					if (dy) {
						viewZoom *= exp((float)dy * -.03f); 
					} 
				} else {
					if (dx || dy) {
						viewPosX -= (float)dx * 0.01f;
						viewPosY += (float)dy * 0.01f;
					}
				}
			}
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
		if (event.button.button == SDL_BUTTON_LEFT) {
			leftButtonDown = true;
		}
		break;
	case SDL_MOUSEBUTTONUP:
		if (event.button.button == SDL_BUTTON_LEFT) {
			leftButtonDown = false;
		}
		break;
	case SDL_KEYDOWN:
		if (event.key.keysym.sym == SDLK_LSHIFT) {
			leftShiftDown = true;
		} else if (event.key.keysym.sym == SDLK_RSHIFT) {
			rightShiftDown = true;
		} else if (event.key.keysym.sym == SDLK_u) {
			if (doUpdate) {
				doUpdate = 0;
			} else {
				if (shiftDown) {
					doUpdate = 2;
				} else {
					doUpdate = 1;
				}
			}
		}
		break;
	case SDL_KEYUP:
		if (event.key.keysym.sym == SDLK_LSHIFT) {
			leftShiftDown = false;
		} else if (event.key.keysym.sym == SDLK_RSHIFT) {
			rightShiftDown = false;
		}
		break;
	}
}

GLAPP_MAIN(HydroGPUApp)


#include "HydroGPU/CLApp.h"
#include "Common/Exception.h"
#include "TensorMath/Vector.h"
#include <iostream>
#include <sstream>

CLApp::CLApp()
: GLApp()
, useGPU(true)
{}

cl::Platform CLApp::getPlatform() {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for(cl::Platform &platform : platforms) {
		std::cout << "platform " << platform() << std::endl;
		std::vector<std::pair<cl_uint, const char *>> queries = {
			std::pair<cl_uint, const char *>(CL_PLATFORM_NAME, "name"),
			std::pair<cl_uint, const char *>(CL_PLATFORM_VENDOR, "vendor"),
			std::pair<cl_uint, const char *>(CL_PLATFORM_VERSION, "version"),
			std::pair<cl_uint, const char *>(CL_PLATFORM_PROFILE, "profile"),
			std::pair<cl_uint, const char *>(CL_PLATFORM_EXTENSIONS, "extensions"),
		};
		for (std::pair<cl_uint, const char *> &query : queries) {
			std::string param;
			platform.getInfo(query.first, &param);
			std::cout << query.second << ":\t" << param << std::endl;
		}
		std::cout << std::endl;
	}

	return platforms[0];
}

struct DeviceParameterQuery {
	cl_uint param;
	const char *name;
	bool failed;
	DeviceParameterQuery(cl_uint param_, const char *name_) : param(param_), name(name_), failed(false) {}
	virtual void query(cl::Device device) = 0;
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
	virtual void query(cl::Device device) {
		try {
			device.getInfo(param, &value);
		} catch (cl::Error &) {
			failed = true;
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
	virtual void query(cl::Device device) {
		try {
			device.getInfo(param, &value);
		} catch (cl::Error &) {
			failed = true;
		}
	}
	virtual std::string toStringType() { return value; }
};

//has to be a class separate of DeviceParameterQueryType because some types are used with both (cl_device_fp_config is typedef'd as a cl_uint)
template<typename Type>
struct DeviceParameterQueryEnumType : public DeviceParameterQuery {
	Type value;
	DeviceParameterQueryEnumType(cl_uint param_, const char *name_) : DeviceParameterQuery(param_, name_), value(Type()) {}
	virtual void query(cl::Device device) {
		try {
			device.getInfo(param, &value);
		} catch (cl::Error &) {
			failed = true;
		}
	}
	virtual std::vector<std::pair<Type, const char *>> getFlags() = 0;
	virtual std::string toStringType() {
		std::vector<std::pair<Type, const char *>>flags = getFlags();
		Type copy = value;
		std::stringstream ss;
		for (std::pair<Type, const char*> &flag : flags) {
			if (copy & flag.first) {
				copy -= flag.first;
				ss << "\n\t" << flag.second;
			}
		}
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

cl::Device CLApp::getDevice(cl::Platform platform) {
	std::vector<cl::Device> devices;
	platform.getDevices(useGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, &devices);

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

	std::vector<cl::Device>::iterator deviceIter =
		std::find_if(devices.begin(), devices.end(), [&](cl::Device device)
	{
		std::cout << "device " << device() << std::endl;
		for (std::shared_ptr<DeviceParameterQuery> query : deviceParameters) {
			query->query(device);
			std::cout << query->name << ":\t" << query->tostring() << std::endl;
		}

		std::string extensionStr = device.getInfo<CL_DEVICE_EXTENSIONS>();
		std::istringstream iss(extensionStr);
		
		std::vector<std::string> extensions;
		std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), std::back_inserter<std::vector<std::string>>(extensions));

		std::cout << "extensions:" << std::endl;
		for (std::string &s : extensions) {
			std::cout << "\t" << s << std::endl;
		}
		std::cout << std::endl;

		std::vector<std::string>::iterator extension = 
			std::find_if(extensions.begin(), extensions.end(), [&](const std::string &s)
		{
			return s == std::string("cl_khr_gl_sharing") 	//i don't have
				|| s == std::string("cl_APPLE_gl_sharing");	//i do have!
		});
	 
		return extension != extensions.end();
	});
	if (deviceIter == devices.end()) throw Exception() << "failed to find a device capable of GL sharing";
	return *deviceIter;
}

void CLApp::init() {
	platform = getPlatform();
	device = getDevice(platform);

#if PLATFORM_osx
	CGLContextObj kCGLContext = CGLGetCurrentContext();	// GL Context
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext); // Share Group
	cl_context_properties properties[] = {
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
		0
	};
#endif
#if PLATFORM_windows
	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), // HGLRC handle
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), // HDC handle
		CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
		0
	};	
#endif
	context = cl::Context({device}, properties);
	commands = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
}


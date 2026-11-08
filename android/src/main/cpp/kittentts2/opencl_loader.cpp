// The OpenCL functions ggml's OpenCL backend calls, forwarded to the phone's own
// libOpenCL.so, which is opened on first use. Linking to libOpenCL.so instead
// would stop the engine loading on phones without one; here they get no OpenCL
// platform and ggml leaves OpenCL out. Written by opencl_loader.py from Khronos'
// CL/cl_function_types.h; regenerate it when ggml calls more of OpenCL.

// As ggml-opencl.cpp has them, so the declarations match.
#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_function_types.h>
#include <dlfcn.h>
#include <initializer_list>

namespace {

void * openclSymbol(const char * name) {
    static void * library = [] () -> void * {
        // An app reaches the vendor's library by name when the device lists it as public
        // (and the app's manifest asks for it, from Android 12); the paths are for older phones.
        for (const char * path : {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so"}) {
            if (void * handle = dlopen(path, RTLD_NOW | RTLD_LOCAL)) return handle;
        }
        return nullptr;
    }();
    return library ? dlsym(library, name) : nullptr;
}

} // namespace

extern "C" {

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
    static const auto fn = reinterpret_cast<clBuildProgram_fn>(openclSymbol("clBuildProgram"));
    return fn ? fn(program, num_devices, device_list, options, pfn_notify, user_data) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateBuffer_fn>(openclSymbol("clCreateBuffer"));
    if (fn) return fn(context, flags, size, host_ptr, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBufferWithProperties(cl_context context, const cl_mem_properties* properties, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateBufferWithProperties_fn>(openclSymbol("clCreateBufferWithProperties"));
    if (fn) return fn(context, properties, flags, size, host_ptr, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateCommandQueue_fn>(openclSymbol("clCreateCommandQueue"));
    if (fn) return fn(context, device, properties, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices, void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data), void* user_data, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateContext_fn>(openclSymbol("clCreateContext"));
    if (fn) return fn(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateImage_fn>(openclSymbol("clCreateImage"));
    if (fn) return fn(context, flags, image_format, image_desc, host_ptr, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateKernel_fn>(openclSymbol("clCreateKernel"));
    if (fn) return fn(program, kernel_name, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const size_t* lengths, const unsigned char** binaries, cl_int* binary_status, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateProgramWithBinary_fn>(openclSymbol("clCreateProgramWithBinary"));
    if (fn) return fn(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateProgramWithSource_fn>(openclSymbol("clCreateProgramWithSource"));
    if (fn) return fn(context, count, strings, lengths, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void* buffer_create_info, cl_int* errcode_ret) {
    static const auto fn = reinterpret_cast<clCreateSubBuffer_fn>(openclSymbol("clCreateSubBuffer"));
    if (fn) return fn(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrierWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    static const auto fn = reinterpret_cast<clEnqueueBarrierWithWaitList_fn>(openclSymbol("clEnqueueBarrierWithWaitList"));
    return fn ? fn(command_queue, num_events_in_wait_list, event_wait_list, event) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    static const auto fn = reinterpret_cast<clEnqueueCopyBuffer_fn>(openclSymbol("clEnqueueCopyBuffer"));
    return fn ? fn(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list, event) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    static const auto fn = reinterpret_cast<clEnqueueFillBuffer_fn>(openclSymbol("clEnqueueFillBuffer"));
    return fn ? fn(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarkerWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    static const auto fn = reinterpret_cast<clEnqueueMarkerWithWaitList_fn>(openclSymbol("clEnqueueMarkerWithWaitList"));
    return fn ? fn(command_queue, num_events_in_wait_list, event_wait_list, event) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    static const auto fn = reinterpret_cast<clEnqueueNDRangeKernel_fn>(openclSymbol("clEnqueueNDRangeKernel"));
    return fn ? fn(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    static const auto fn = reinterpret_cast<clEnqueueReadBuffer_fn>(openclSymbol("clEnqueueReadBuffer"));
    return fn ? fn(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    static const auto fn = reinterpret_cast<clEnqueueWriteBuffer_fn>(openclSymbol("clEnqueueWriteBuffer"));
    return fn ? fn(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue command_queue) {
    static const auto fn = reinterpret_cast<clFinish_fn>(openclSymbol("clFinish"));
    return fn ? fn(command_queue) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clFlush(cl_command_queue command_queue) {
    static const auto fn = reinterpret_cast<clFlush_fn>(openclSymbol("clFlush"));
    return fn ? fn(command_queue) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {
    static const auto fn = reinterpret_cast<clGetDeviceIDs_fn>(openclSymbol("clGetDeviceIDs"));
    return fn ? fn(platform, device_type, num_entries, devices, num_devices) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    static const auto fn = reinterpret_cast<clGetDeviceInfo_fn>(openclSymbol("clGetDeviceInfo"));
    return fn ? fn(device, param_name, param_value_size, param_value, param_value_size_ret) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    static const auto fn = reinterpret_cast<clGetEventProfilingInfo_fn>(openclSymbol("clGetEventProfilingInfo"));
    return fn ? fn(event, param_name, param_value_size, param_value, param_value_size_ret) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    static const auto fn = reinterpret_cast<clGetKernelInfo_fn>(openclSymbol("clGetKernelInfo"));
    return fn ? fn(kernel, param_name, param_value_size, param_value, param_value_size_ret) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelSubGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name, size_t input_value_size, const void* input_value, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    static const auto fn = reinterpret_cast<clGetKernelSubGroupInfo_fn>(openclSymbol("clGetKernelSubGroupInfo"));
    return fn ? fn(kernel, device, param_name, input_value_size, input_value, param_value_size, param_value, param_value_size_ret) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    static const auto fn = reinterpret_cast<clGetKernelWorkGroupInfo_fn>(openclSymbol("clGetKernelWorkGroupInfo"));
    return fn ? fn(kernel, device, param_name, param_value_size, param_value, param_value_size_ret) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    static const auto fn = reinterpret_cast<clGetPlatformIDs_fn>(openclSymbol("clGetPlatformIDs"));
    return fn ? fn(num_entries, platforms, num_platforms) : CL_PLATFORM_NOT_FOUND_KHR;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    static const auto fn = reinterpret_cast<clGetPlatformInfo_fn>(openclSymbol("clGetPlatformInfo"));
    return fn ? fn(platform, param_name, param_value_size, param_value, param_value_size_ret) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    static const auto fn = reinterpret_cast<clGetProgramBuildInfo_fn>(openclSymbol("clGetProgramBuildInfo"));
    return fn ? fn(program, device, param_name, param_value_size, param_value, param_value_size_ret) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    static const auto fn = reinterpret_cast<clGetProgramInfo_fn>(openclSymbol("clGetProgramInfo"));
    return fn ? fn(program, param_name, param_value_size, param_value, param_value_size_ret) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent(cl_event event) {
    static const auto fn = reinterpret_cast<clReleaseEvent_fn>(openclSymbol("clReleaseEvent"));
    return fn ? fn(event) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
    static const auto fn = reinterpret_cast<clReleaseKernel_fn>(openclSymbol("clReleaseKernel"));
    return fn ? fn(kernel) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj) {
    static const auto fn = reinterpret_cast<clReleaseMemObject_fn>(openclSymbol("clReleaseMemObject"));
    return fn ? fn(memobj) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program program) {
    static const auto fn = reinterpret_cast<clReleaseProgram_fn>(openclSymbol("clReleaseProgram"));
    return fn ? fn(program) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {
    static const auto fn = reinterpret_cast<clSetKernelArg_fn>(openclSymbol("clSetKernelArg"));
    return fn ? fn(kernel, arg_index, arg_size, arg_value) : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents(cl_uint num_events, const cl_event* event_list) {
    static const auto fn = reinterpret_cast<clWaitForEvents_fn>(openclSymbol("clWaitForEvents"));
    return fn ? fn(num_events, event_list) : CL_INVALID_OPERATION;
}

} // extern "C"

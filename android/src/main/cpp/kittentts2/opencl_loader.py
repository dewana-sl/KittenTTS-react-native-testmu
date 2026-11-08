"""Writes opencl_loader.cpp from Khronos' CL/cl_function_types.h:

    python3 opencl_loader.py <OpenCL-Headers>/CL/cl_function_types.h opencl_loader.cpp

NAMES are the OpenCL functions ggml's OpenCL backend calls; add any a newer ggml
calls (linking fails, naming them).
"""
import re, sys

NAMES = """
clBuildProgram clCreateBuffer clCreateBufferWithProperties clCreateCommandQueue clCreateContext
clCreateImage clCreateKernel clCreateProgramWithBinary clCreateProgramWithSource clCreateSubBuffer
clEnqueueBarrierWithWaitList clEnqueueCopyBuffer clEnqueueFillBuffer clEnqueueMarkerWithWaitList
clEnqueueNDRangeKernel clEnqueueReadBuffer clEnqueueWriteBuffer clFinish clFlush clGetDeviceIDs
clGetDeviceInfo clGetEventProfilingInfo clGetKernelInfo clGetKernelSubGroupInfo clGetKernelWorkGroupInfo
clGetPlatformIDs clGetPlatformInfo clGetProgramBuildInfo clGetProgramInfo clReleaseEvent clReleaseKernel
clReleaseMemObject clReleaseProgram clSetKernelArg clWaitForEvents
""".split()

header, out = sys.argv[1], sys.argv[2]
names = NAMES
text = open(header).read()
functions = []
for name in names:
    m = re.search(r"typedef\s+([\w\s\*]+?)\s+CL_API_CALL\s+" + name + r"_t\(\s*(.*?)\);", text, re.S)
    if not m:
        sys.exit("no typedef for " + name)
    ret = " ".join(m.group(1).split())
    # Split at the commas outside parentheses: callbacks have their own parameter lists.
    params, depth, current = [], 0, ""
    for ch in m.group(2):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            params.append(current)
            current = ""
        else:
            current += ch
    params = [" ".join(p.split()) for p in params + [current]]
    if params == ["void"]:
        params = []
    def arg_name(p):
        callback = re.search(r"\(\s*CL_CALLBACK\s*\*\s*(\w+)\s*\)", p)
        return callback.group(1) if callback else re.search(r"(\w+)(\s*\[\s*\])?$", p).group(1)
    args = [arg_name(p) for p in params]
    functions.append((name, ret, params, args))

lines = []
w = lines.append
w("// The OpenCL functions ggml's OpenCL backend calls, forwarded to the phone's own")
w("// libOpenCL.so, which is opened on first use. Linking to libOpenCL.so instead")
w("// would stop the engine loading on phones without one; here they get no OpenCL")
w("// platform and ggml leaves OpenCL out. Written by opencl_loader.py from Khronos'")
w("// CL/cl_function_types.h; regenerate it when ggml calls more of OpenCL.")
w("")
w("// As ggml-opencl.cpp has them, so the declarations match.")
w("#define CL_TARGET_OPENCL_VERSION 300")
w("#define CL_USE_DEPRECATED_OPENCL_1_2_APIS")
w("")
w("#include <CL/cl.h>")
w("#include <CL/cl_ext.h>")
w("#include <CL/cl_function_types.h>")
w("#include <dlfcn.h>")
w("#include <initializer_list>")
w("")
w("namespace {")
w("")
w("void * openclSymbol(const char * name) {")
w("    static void * library = [] () -> void * {")
w("        // An app reaches the vendor's library by name when the device lists it as public")
w("        // (and the app's manifest asks for it, from Android 12); the paths are for older phones.")
w("        for (const char * path : {\"libOpenCL.so\", \"/vendor/lib64/libOpenCL.so\", \"/system/vendor/lib64/libOpenCL.so\"}) {")
w("            if (void * handle = dlopen(path, RTLD_NOW | RTLD_LOCAL)) return handle;")
w("        }")
w("        return nullptr;")
w("    }();")
w("    return library ? dlsym(library, name) : nullptr;")
w("}")
w("")
w("} // namespace")
w("")
w("extern \"C\" {")
for name, ret, params, args in functions:
    w("")
    w(f"CL_API_ENTRY {ret} CL_API_CALL {name}({', '.join(params)}) {{")
    w(f"    static const auto fn = reinterpret_cast<{name}_fn>(openclSymbol(\"{name}\"));")
    call = f"fn({', '.join(args)})"
    if ret == "cl_int":
        missing = "CL_PLATFORM_NOT_FOUND_KHR" if name == "clGetPlatformIDs" else "CL_INVALID_OPERATION"
        w(f"    return fn ? {call} : {missing};")
    elif "errcode_ret" in args:
        w(f"    if (fn) return {call};")
        w("    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;")
        w("    return nullptr;")
    else:
        sys.exit(f"{name}: what to return without the library? ({ret})")
    w("}")
w("")
w("} // extern \"C\"")
open(out, "w").write("\n".join(lines) + "\n")
print(f"{len(functions)} functions")

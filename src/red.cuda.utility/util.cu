// includes system
#include <ostream>
#include <string>

// includes project
#include "util.h"
#include "red_type.h"
#include "red_macro.h"

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions

int device_query(ostream& sout, int id_dev)
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        exit(EXIT_FAILURE);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, id_dev);

    sout << "The code runs on " << deviceProp.name << " device:" << endl;

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
	sout << "  CUDA Driver Version / Runtime Version          " << driverVersion/1000 << "." << (driverVersion%100)/10 << " / " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << endl;
	sout << "  CUDA Capability Major/Minor version number:    " << deviceProp.major << "." << deviceProp.minor << endl;
	sout << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem/1048576.0f << " MBytes" << endl;
	sout << "  " << deviceProp.multiProcessorCount <<  " Multiprocessors, " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << " CUDA Cores/MP:     " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount << " CUDA Cores" << endl;
	sout << "  GPU Clock rate:                                " << deviceProp.clockRate * 1e-3f << ".0 MHz" << endl;
	sout << "  Total amount of constant memory:               " << deviceProp.totalConstMem << " bytes" << endl;
	sout << "  Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << " bytes" << endl;
	sout << "  Total number of registers available per block: " << deviceProp.regsPerBlock << endl;
	sout << "  Warp size:                                     " << deviceProp.warpSize << endl;
	sout << "  Maximum number of threads per multiprocessor:  " << deviceProp.maxThreadsPerMultiProcessor << endl;
	sout << "  Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock << endl;
	sout << "  Max dimension size of a thread block (x,y,z): (" << deviceProp.maxThreadsDim[0] << "," << deviceProp.maxThreadsDim[1] << "," << deviceProp.maxThreadsDim[2] << ")" << endl;
	sout << "  Max dimension size of a grid size    (x,y,z): (" <<deviceProp.maxGridSize[0] << "," << deviceProp.maxGridSize[1] << "," << deviceProp.maxGridSize[2] << ")" << endl << endl;

    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;

    // Print Out all device Names
    for (dev = 0; dev < deviceCount; ++dev)
    {
#ifdef _WIN32
        sprintf_s(cTemp, 13, ", Device%d = ", dev);
#else
        sprintf(cTemp, ", Device%d = ", dev);
#endif
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        sProfileString += cTemp;
        sProfileString += deviceProp.name;
    }
    sProfileString += "\n";

	sout << sProfileString;

    // finish
    return (EXIT_SUCCESS);
}

void allocate_device_vector(void **d_ptr, size_t size, const char *file, int line)
{
	cudaMalloc(d_ptr, size);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMalloc failed (allocate_device_vector)");
	}
}

void copy_vector_to_device(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemcpy failed (copy_vector_to_device)");
	}
}

void copy_vector_to_host(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemcpy failed (copy_vector_to_host)");
	}
}

void copy_vector_d2d(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemcpy failed (copy_vector_d2d)");
	}
}

void copy_constant_to_device(const void* dst, const void *src, size_t count)
{
	cudaMemcpyToSymbol(dst, src, count, 0, cudaMemcpyHostToDevice);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemcpyToSymbol failed (copy_constant_to_device)");
	}
}

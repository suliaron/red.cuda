// includes system
#include <iomanip>
#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>

// includes project
#include "util.h"
#include "file_util.h"
#include "red_type.h"
#include "red_macro.h"

namespace redutilcu
{

template <typename T>
std::string number_to_string( T number )
{
	std::ostringstream ss;
	ss << number;
	return ss.str();
}

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

void allocate_host_vector(void **ptr, size_t size, const char *file, int line)
{
	*ptr = (void *)malloc(size);
	if (0x0 == ptr)
	{
		throw string("malloc failed (allocate_host_vector)");
	}

	// Clear memory 
	memset(*ptr, 0, size);
}

void allocate_device_vector(void **ptr, size_t size, const char *file, int line)
{
	cudaMalloc(ptr, size);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMalloc failed (allocate_device_vector)");
	}

	// Clear memory 
	cudaMemset(*ptr, 0, size);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemset failed (allocate_device_vector)");
	}
}

void allocate_vector(void **ptr, size_t size, bool cpu, const char *file, int line)
{
	if (cpu)
	{
		allocate_host_vector(ptr, size, file, line);
	}
	else
	{
		allocate_device_vector(ptr, size, file, line);
	}
}

void free_host_vector(void **ptr, const char *file, int line)
{
	if (0x0 != *ptr)
	{
		delete[] *ptr;
		*ptr = (void *)0x0;
	}
}

void free_device_vector(void **ptr, const char *file, int line)
{
	if (0x0 != *ptr)
	{
		cudaFree(*ptr);
		cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("cudaFree failed (free_device_vector)");
		}
		*ptr = (void *)0x0;
	}
}

void free_vector(void **ptr, bool cpu, const char *file, int line)
{
	if (cpu)
	{
		free_host_vector(ptr, file, line);
	}
	else
	{
		free_device_vector(ptr, file, line);
	}
}

void allocate_host_storage(sim_data_t *sd, int n)
{
	sd->h_y.resize(2);
	sd->h_yout.resize(2);

	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_HOST_VECTOR((void **)&(sd->h_y[i]),    n*sizeof(vec_t));
		ALLOCATE_HOST_VECTOR((void **)&(sd->h_yout[i]), n*sizeof(vec_t));
	}
	ALLOCATE_HOST_VECTOR((void **)&(sd->h_p),           n*sizeof(param_t));
	ALLOCATE_HOST_VECTOR((void **)&(sd->h_body_md),     n*sizeof(body_metadata_t));
	ALLOCATE_HOST_VECTOR((void **)&(sd->h_epoch),       n*sizeof(ttt_t));

	ALLOCATE_HOST_VECTOR((void **)&(sd->h_oe),          n*sizeof(orbelem_t));
}

void allocate_device_storage(sim_data_t *sd, int n)
{
	sd->d_y.resize(2);
	sd->d_yout.resize(2);

	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void **)&(sd->d_y[i]),	  n*sizeof(vec_t));
		ALLOCATE_DEVICE_VECTOR((void **)&(sd->d_yout[i]), n*sizeof(vec_t));
	}
	ALLOCATE_DEVICE_VECTOR((void **)&(sd->d_p),			  n*sizeof(param_t));
	ALLOCATE_DEVICE_VECTOR((void **)&(sd->d_body_md),	  n*sizeof(body_metadata_t));
	ALLOCATE_DEVICE_VECTOR((void **)&(sd->d_epoch),		  n*sizeof(ttt_t));

    ALLOCATE_DEVICE_VECTOR((void **)&(sd->d_oe),          n*sizeof(orbelem_t));
}

void deallocate_host_storage(sim_data_t *sd)
{
	for (int i = 0; i < 2; i++)
	{
		FREE_HOST_VECTOR((void **)&(sd->h_y[i]));
		FREE_HOST_VECTOR((void **)&(sd->h_yout[i]));
	}
	FREE_HOST_VECTOR((void **)&(sd->h_p));
	FREE_HOST_VECTOR((void **)&(sd->h_body_md));
	FREE_HOST_VECTOR((void **)&(sd->h_epoch));

	FREE_HOST_VECTOR((void **)&(sd->h_oe));
}

void deallocate_device_storage(sim_data_t *sd)
{
	for (int i = 0; i < 2; i++)
	{
		FREE_DEVICE_VECTOR((void **)&(sd->d_y[i]));
		FREE_DEVICE_VECTOR((void **)&(sd->d_yout[i]));
	}
	FREE_DEVICE_VECTOR((void **)&(sd->d_p));
	FREE_DEVICE_VECTOR((void **)&(sd->d_body_md));
	FREE_DEVICE_VECTOR((void **)&(sd->d_epoch));

    FREE_DEVICE_VECTOR((void **)&(sd->d_oe));
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

int get_id_fastest_GPU()
{
	// TODO: implement
	std::cerr << "       TODO: implement int get_id_fastest_GPU() function!" << endl;

	return 0;
}

void set_device(int id_of_target_dev, bool verbose)
{
	cudaError_t cudaStatus = cudaSuccess;

	int n_device = 0;
	cudaGetDeviceCount(&n_device);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaGetDeviceCount() failed");
	}
	if (0 == n_device)
	{
		throw string("No CUDA device was found. ");
	}

	if (verbose)
	{
		file::log_message(std::cout, "The number of CUDA device(s): " + number_to_string<int>(n_device));
	}
	if (n_device > id_of_target_dev && 0 <= id_of_target_dev)
	{
		// Set the desired id of the device
		cudaSetDevice(id_of_target_dev);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("cudaSetDevice() failed");
		}
		if (verbose)
		{
			file::log_message(std::cout, "Execution will be transferred to the GPU with id: " + number_to_string<int>(id_of_target_dev));
		}
	}
	else
	{
		throw string("The device with the requested id does not exist!");
	}
}

void print_array(string path, int n, var_t *data, computing_device_t comp_dev)
{
	var_t* h_data = 0x0;

	ostream *out = 0x0;
	if (0 < path.length())
	{
		out = new ofstream(path.c_str(), ios::app);
	}
	else
	{
		out = &cout;
	}

	out->setf(ios::right);
	out->setf(ios::scientific);

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		h_data = new var_t[n];
		copy_vector_to_host(h_data, data, n * sizeof(var_t));
	}
	else
	{
		h_data = data;
	}
	for (int i = 0; i < n; i++)
	{
		*out << setw(5) << i << setprecision(16) << setw(25) << h_data[i] << endl;
	}

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		delete[] h_data;
	}
	if (0 < path.length())
	{
		out->flush();
		delete out;
	}
}

void create_aliases(computing_device_t comp_dev, sim_data_t *sd)
{
	switch (comp_dev)
	{
	case COMPUTING_DEVICE_CPU:
		for (int i = 0; i < 2; i++)
		{
			sd->y[i]    = sd->h_y[i];
			sd->yout[i] = sd->h_yout[i];
		}
		sd->p       = sd->h_p;
		sd->body_md = sd->h_body_md;
		sd->epoch   = sd->h_epoch;
        sd->oe      = sd->h_oe;
		break;
	case COMPUTING_DEVICE_GPU:
		for (int i = 0; i < 2; i++)
		{
			sd->y[i]    = sd->d_y[i];
			sd->yout[i] = sd->d_yout[i];
		}
		sd->p       = sd->d_p;
		sd->body_md = sd->d_body_md;
		sd->epoch   = sd->d_epoch;
        sd->oe      = sd->d_oe;
		break;
	}
}

} /* redutilcu */

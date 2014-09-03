// includes system
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>

// includes CUDA
#include "cuda.h"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// includes project
#include "int_euler.h"
#include "int_rungekutta4.h"
#include "parameter.h"
#include "redutilcu.h"
#include "nbody_exception.h"
#include "options.h"
#include "red_type.h"

using namespace std;
using namespace redutilcu;

int device_query(int argc, const char **argv)
{
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
	}

    printf("\n");
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
    printf("%s", sProfileString.c_str());

	printf("Result = PASS\n");

    // finish
    return (EXIT_SUCCESS);
}

int main(int argc, const char** argv)
{
	cout << tools::get_time_stamp() << endl;
	printf("%s Starting...\n\n", argv[0]);

	device_query(argc, argv);

	time_t start = time(NULL);
	var_t sum_time_of_steps = 0.0;
	int_t n_step = 0;
	try
	{
		options opt = options(argc, argv);
		pp_disk *ppd = opt.create_pp_disk();

		integrator::euler *intgr = new integrator::euler(ppd, 0.001);
		//integrator::rungekutta4 *intgr = new integrator::rungekutta4(opt.param->start_time, 0.1, opt.param->adaptive, opt.param->tolerance, ppd);

		ttt_t ps			= 0;
		ttt_t dt			= 0;
		string path = file::combine_path(opt.printout_dir, "result.out.txt");
		ostream* result_f = new ofstream(path.c_str(), ios::out);
		//path = file::combine_path(opt.printout_dir, "event.out.txt");
		//ostream* event_f = new ofstream(path.c_str(), ios::out);
		//path = file::combine_path(opt.printout_dir, "log.txt");
		//ostream* log_f = new ofstream(path.c_str(), ios::out);

		ppd->print_body_data(*result_f);
		while (ppd->t <= opt.param->stop_time)
		{
			clock_t start_of_step = clock();
			dt = intgr->step();
			clock_t end_of_step = clock();
			sum_time_of_steps += (end_of_step - start_of_step);
			n_step++;
			if (n_step % 1000 == 0) 
			{
				cout << "Time for one step: " << (end_of_step - start_of_step) / (double)CLOCKS_PER_SEC << " s, avg: " << sum_time_of_steps / (double)CLOCKS_PER_SEC / n_step << " s" << endl;
			}

			ps += fabs(dt);
			if (fabs(ps) >= opt.param->output_interval)
			{
				ps = 0.0;
				ppd->call_kernel_transform_to(0);
				ppd->copy_to_host();
				ppd->print_body_data(*result_f);
			}
		} /* while */
		ppd->copy_to_host();
		ppd->print_body_data(*result_f);

	} /* try */
	catch (const nbody_exception& ex)
	{
		cerr << "Error: " << ex.what() << endl;
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}
	cout << "Total time: " << time(NULL) - start << " s" << endl;


    return (EXIT_SUCCESS);
}

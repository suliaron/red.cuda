// includes system
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>

// includes CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "options.h"

using namespace std;

int main(int argc, const char** argv, const char** env)
{
	time_t start = time(NULL);

	ofstream* output[OUTPUT_NAME_N];
	memset(output, 0x0, sizeof(output));

	try
	{
		options opt = options(argc, argv);
	} /* try */
	catch (const string& msg)
	{
		//if (0x0 != output[OUTPUT_NAME_LOG])
		//{
		//	file::log_message(*output[OUTPUT_NAME_LOG], "Error: " + msg, false);
		//}
		cerr << "Error: " << msg << endl;
	}

	//if (0x0 != output[OUTPUT_NAME_LOG])
	//{
	//	file::log_message(*output[OUTPUT_NAME_LOG], "Total time: " + tools::convert_time_t(time(NULL) - start) + " s", false);
	//}
	cout << "Total time: " << time(NULL) - start << " s" << endl;

	return (EXIT_SUCCESS);
}

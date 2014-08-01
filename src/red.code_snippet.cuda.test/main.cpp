// includes system
#include <iostream>
#include <string>

// includes project
#include "red_type.h"

using namespace std;

int main(int argc, const char** argv)
{
	var_t xmax =  1.0;
	var_t xmin = -1.0;

	sim_data_t sim_data;

	sim_data.pos = new posm_t[8];

	for (int i = 0; i < 8; i++)
	{
		sim_data.pos[i].x = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		sim_data.pos[i].y = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		sim_data.pos[i].z = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		sim_data.pos[i].m = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
	}

	for (int i = 0; i < 8; i++)
	{
		printf("address of sim_data.pos[i]: %p\n", &sim_data.pos[i]);
		printf("\taddress of sim_data.pos[i].x: %p\n", &sim_data.pos[i].x);
		printf("\taddress of sim_data.pos[i].y: %p\n", &sim_data.pos[i].y);
		printf("\taddress of sim_data.pos[i].z: %p\n", &sim_data.pos[i].z);
		printf("\taddress of sim_data.pos[i].m: %p\n", &sim_data.pos[i].m);
	}

	sim_data.body_md = new body_metadata_t[8];

	for (int i = 0; i < 8; i++)
	{
		printf("address of sim_data.body_md[i]: %p\n", &sim_data.body_md[i]);
		printf("\taddress of sim_data.body_md[i].id       : %p\n", &sim_data.body_md[i].id);
		printf("\taddress of sim_data.body_md[i].active   : %p\n", &sim_data.body_md[i].active);
		printf("\taddress of sim_data.body_md[i].body_type: %p\n", &sim_data.body_md[i].body_type);
		printf("\taddress of sim_data.body_md[i].mig_type : %p\n", &sim_data.body_md[i].mig_type);
	}

	delete[] sim_data.pos;

	return 0;
}

# if 0
/*
 *  Basic example of an exception
 */
int main(int argc, const char** argv)
{
	try
	{
		throw string("This is an exception!\n");
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}
}
#endif
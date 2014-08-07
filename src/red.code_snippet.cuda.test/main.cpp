// includes system
#include <iostream>
#include <string>

// includes project
#include "red_type.h"

using namespace std;

/*
 * Pointer teszt: 
 * A memóriában egy struktúra adattagjai folytonosan helyezkednek el.
 * Ha egy ilyen struktúrából a new operatorral tömböt hozok létre, akkor a tömb
 * elemei is folytonosan fognak elhelyezkedni.
 * struct my_4double DOES NOT get aligned to 16 bytes. 
 * DE
 * a struct __builtin_align__(16) my_4double_a igen.
 */

struct my_4double
{
	double x, y, z, w;
};

struct __builtin_align__(16) my_4double_a
{
	double x, y, z, w;
};


int main(int argc, const char** argv)
{
	var_t xmax =  1.0;
	var_t xmin = -1.0;

	sim_data_t sim_data;

	{
		posm_t pos;
		printf("address of pos: %p\n", &pos);
		printf("address of pos.x: %p\n", &pos.x);
		printf("address of pos.y: %p\n", &pos.y);
		printf("address of pos.z: %p\n", &pos.z);
		printf("address of pos.m: %p\n", &pos.m);
	}

	{
		struct my_4double pos;
		printf("address of pos: %p\n", &pos);
		printf("address of pos.x: %p\n", &pos.x);
		printf("address of pos.y: %p\n", &pos.y);
		printf("address of pos.z: %p\n", &pos.z);
		printf("address of pos.m: %p\n", &pos.w);

		struct my_4double_a apos;
		printf("address of apos: %p\n", &apos);
		printf("address of apos.x: %p\n", &apos.x);
		printf("address of apos.y: %p\n", &apos.y);
		printf("address of apos.z: %p\n", &apos.z);
		printf("address of apos.m: %p\n", &apos.w);
	}

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
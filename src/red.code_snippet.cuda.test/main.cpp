// includes system
#include <iomanip>
#include <iostream>
#include <string>

// includes project
#include "red_type.h"

using namespace std;

#if 0
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
		vec_t r;
		printf("address of r: %p\n", &r);
		printf("address of r.x: %p\n", &r.x);
		printf("address of r.y: %p\n", &r.y);
		printf("address of r.z: %p\n", &r.z);
		printf("address of r.w: %p\n", &r.w);
	}

	{
		struct my_4double r;
		printf("address of r: %p\n", &r);
		printf("address of r.x: %p\n", &r.x);
		printf("address of r.y: %p\n", &r.y);
		printf("address of r.z: %p\n", &r.z);
		printf("address of r.w: %p\n", &r.w);

		struct my_4double_a _r;
		printf("address of _r: %p\n", &_r);
		printf("address of _r.x: %p\n", &_r.x);
		printf("address of _r.y: %p\n", &_r.y);
		printf("address of _r.z: %p\n", &_r.z);
		printf("address of _r.w: %p\n", &_r.w);
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
#endif

#if 0
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

#if 1

int main(int argc, const char** argv)
{
	var_t a = -123456789.0123456789;
	var_t b =  1.234567890123456789;

	cout.precision(16);
	cout.setf(ios::right);
	cout.setf(ios::scientific);

	cout << "0        1         2         3         4         5" << endl;
	cout << "12345678901234567890123456789012345678901234567890" << endl;
	cout << setw(25) << a << setw(25) << b << endl;
	cout << setw(25) << b << setw(25) << a << endl;

	return 0;
}

#endif
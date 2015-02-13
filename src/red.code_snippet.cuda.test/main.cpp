// includes system
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

// includes project
#include "red_type.h"
#include "red_constants.h"

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

#if 0

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

#if 0
/*
 *  Detect collisions from events array 
 *  Working hypothesis: collisions happen only between two bodies.
 */
typedef struct survivor
	{
		vec_t			r;
		vec_t			v;
		param_t			p;
		body_metadata_t body_md;
	} survivor_t;

int event_counter = 10;

event_data_t* events;			//!< Vector on the host containing data for events (one colliding pair multiple occurances)
vector<event_data_t> sp_events;	//!< Vector on the host containing data for events but  (one colliding pair one occurances)
vector<survivor_t>	survivors;	//!< Vector on the host containing data for the survivor body

void create_sp_events()
{
	bool *processed = new bool[event_counter];
	sp_events.resize(event_counter);

	for (int i = 0; i < event_counter; i++)
	{
		processed[i] = false;
	}

	int n = 0;
	for (int k = 0; k < event_counter; k++)
	{
		if (processed[k] == false)
		{
			processed[k] = true;
			sp_events[n] = events[k];
		}
		else
		{
			continue;
		}
		for (int i = k + 1; i < event_counter; i++)
		{
			if (sp_events[n].id1 == events[i].id1 && sp_events[n].id2 == events[i].id2)
			{
				processed[i] = true;
				if (sp_events[n].t > events[i].t)
				{
					sp_events[n] = events[i];
				}
			}
		}
		n++;
	}

	sp_events.resize(n);
	survivors.resize(n);
}

int main(int argc, char **argv[])
{
	{
		var_t f = constants::GramPerCm2ToSolarPerAu2;
		cout << "1 g/cm2 = 1 " << f << " M_s/AU2" << endl;
		return 0;
	}

	events = new event_data_t[event_counter];

	int i, j, k, l, m, n;

	i =   1, j =   2;
	k =  10, l =  12;
	m = 100, n = 102;

	events[0].event_name = EVENT_NAME_COLLISION;
	events[0].t = 0.3;
	events[0].id1 = i;
	events[0].id2 = j;

	events[1].event_name = EVENT_NAME_COLLISION;
	events[1].t = 0.1;
	events[1].id1 = i;
	events[1].id2 = j;

	events[2].event_name = EVENT_NAME_COLLISION;
	events[2].t = 0.35;
	events[2].id1 = k;
	events[2].id2 = l;

	events[3].event_name = EVENT_NAME_COLLISION;
	events[3].t = 0.32;
	events[3].id1 = i;
	events[3].id2 = j;

	events[4].event_name = EVENT_NAME_COLLISION;
	events[4].t = 0.254;
	events[4].id1 = k;
	events[4].id2 = l;

	events[5].event_name = EVENT_NAME_COLLISION;
	events[5].t = 0.5634;
	events[5].id1 = i;
	events[5].id2 = j;

	events[6].event_name = EVENT_NAME_COLLISION;
	events[6].t = 0.342324;
	events[6].id1 = m;
	events[6].id2 = n;

	events[7].event_name = EVENT_NAME_COLLISION;
	events[7].t = 0.5678901;
	events[7].id1 = k;
	events[7].id2 = l;

	events[8].event_name = EVENT_NAME_COLLISION;
	events[8].t = 0.1234;
	events[8].id1 = i;
	events[8].id2 = j;

	events[9].event_name = EVENT_NAME_COLLISION;
	events[9].t = 0.1234;
	events[9].id1 = m;
	events[9].id2 = n;

	create_sp_events();

	delete[] events;
}

#endif

#if 1

int main()
{
	int r = 5;

	const string err_msg = "rk8_kernel::calc_ytemp_for_f"; 
	ostringstream convert;	// stream used for the conversion
	convert << r;
	cout << err_msg + convert.str() + " failed" << endl;

	return 0;
}

#endif
// includes system
#include <iostream>
#include <string>
#include <fstream>

// includes project
#include "gas_disk.h"
#include "pp_disk.h"
#include "redutilcu.h"

using namespace std;
using namespace redutilcu;

pp_disk::pp_disk(string& path, gas_disk *gd)
{
	get_number_of_bodies(path);
	allocate_storage();
	load(path);
	g_disk = gd;
}

void pp_disk::get_number_of_bodies(string& path)
{
	ifstream input(path.c_str());
	if (input) 
	{
		int ns, ngp, nrp, npp, nspl, npl, ntp;
		ns = ngp = nrp = npp = nspl = npl = ntp = 0;
		input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;
		n_bodies = new number_of_bodies(ns, ngp, nrp, npp, nspl, npl, ntp);
	}
	else 
	{
		throw string("Cannot open " + path + ".");
	}
	input.close();
}

void pp_disk::allocate_storage()
{
	sim_data.pos = new posm_t[n_bodies->total];
	sim_data.vel = new velR_t[n_bodies->total];
	sim_data.params = new param_t[n_bodies->total];
	sim_data.body_md = new body_metadata_t[n_bodies->total];
	sim_data.epoch = new ttt_t[n_bodies->total];
}

var_t pp_disk::get_mass_of_star()
{
	body_metadata_t* body_md = sim_data.body_md;
	for (int j = 0; j < n_bodies->n_massive(); j++ ) {
		if (body_md[j].body_type == BODY_TYPE_STAR)
		{
			return sim_data.params[j].mass;
		}
	}
	throw string("No star is included!");
}

var_t pp_disk::get_total_mass()
{
	var_t totalMass = 0.0;

	param_t* param = sim_data.params;
	for (int j = 0; j < n_bodies->n_massive(); j++ ) {
		totalMass += param[j].mass;
	}

	return totalMass;
}

void pp_disk::compute_bc(posm_t* R0, velR_t* V0)
{
	posm_t* coor = sim_data.pos;
	velR_t* velo = sim_data.vel;

	for (int j = 0; j < n_bodies->n_massive(); j++ ) {
		R0->x += coor[j].m * coor[j].x;
		R0->y += coor[j].m * coor[j].y;
		R0->z += coor[j].m * coor[j].z;

		V0->x += coor[j].m * velo[j].x;
		V0->y += coor[j].m * velo[j].y;
		V0->z += coor[j].m * velo[j].z;
	}
	var_t M0 = get_total_mass();

	R0->x /= M0;	R0->y /= M0;	R0->z /= M0;
	V0->x /= M0;	V0->y /= M0;	V0->z /= M0;
}

void pp_disk::transform_to_bc()
{
	cout << "Transforming to barycentric system ... ";

	// Position and velocity of the system's barycenter
	posm_t R0 = {0.0, 0.0, 0.0, 0.0};
	velR_t V0 = {0.0, 0.0, 0.0, 0.0};

	compute_bc(&R0, &V0);

	posm_t* coor = sim_data.pos;
	velR_t* velo = sim_data.vel;
	// Transform the bodies coordinates and velocities
	for (int j = 0; j < n_bodies->n_total(); j++ ) {
		coor[j].x -= R0.x;		coor[j].y -= R0.y;		coor[j].z -= R0.z;
		velo[j].x -= V0.x;		velo[j].y -= V0.y;		velo[j].z -= V0.z;
	}

	cout << "done" << endl;
}

void pp_disk::load(string& path)
{
	cout << "Loading " << path << " ... ";

	ifstream input(path.c_str());
	if (input) 
	{
		int ns, ngp, nrp, npp, nspl, npl, ntp;
		input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;
	}
	else 
	{
		throw string("Cannot open " + path + ".");
	}

	posm_t* coor = sim_data.pos;
	velR_t* velo = sim_data.vel;
	param_t* param = sim_data.params;
	body_metadata_t* body_md = sim_data.body_md;
	ttt_t* epoch = sim_data.epoch;

	if (input) {
		int_t	type = 0;
		var_t	cd = 0.0;
		string	dummy;
        		
		for (int i = 0; i < n_bodies->total; i++) { 
			body_md[i].active = true;
			// id
			input >> body_md[i].id;
			// name
			input >> dummy;
			body_names.push_back(dummy);
			// body type
			input >> type;
			body_md[i].body_type = static_cast<body_type_t>(type);
			// epoch
			input >> epoch[i];

			// mass
			input >> param[i].mass;
			coor[i].m = param[i].mass;
			// radius
			input >> velo[i].R;
			// density
			input >> param[i].density;
			// stokes constant
			input >> param[i].cd;

			// migration type
			input >> type;
			body_md[i].mig_type = static_cast<migration_type_t>(type);
			// migration stop at
			input >> param[i].mig_stop_at;

			// position
			input >> coor[i].x;
			input >> coor[i].y;
			input >> coor[i].z;
			// velocity
			input >> velo[i].x;
			input >> velo[i].y;
			input >> velo[i].z;
        }
        input.close();
	}
	else {
		throw string("Cannot open " + path + ".");
	}

	cout << "done" << endl;
}


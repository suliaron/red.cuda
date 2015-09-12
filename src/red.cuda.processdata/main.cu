// includes, system 
#include <stdint.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <ostream>
#include <sstream>

// includes, project
#include "red_constants.h"
#include "red_type.h"
#include "redutilcu.h"


using namespace std;
using namespace redutilcu;

vector<string> body_names;

vector<ttt_t> g_epoch;
vector<body_metadata_t> g_bmd;
vector<param_t> g_param;
vector<vec_t> g_coor;
vector<vec_t> g_velo;

unsigned int ns, ngp, nrp, npp, nspl, npl, ntp, n_total;

void load_body_record(ifstream& input, int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v);

void print(string &path, sim_data_t *sd)
{
	printf("Writing %s to disk .", path.c_str());

	ofstream output(path.c_str(), ios_base::out);
	if (output)
	{
		int_t nBodies = n_total;

		int p = 1;
		for (int i = 0; i < nBodies; i++)
		{
			if (p <= (int)((((var_t)i/(var_t)nBodies))*100.0))
			{
				printf(".");
				p++;
			}
			file::print_body_record(output, body_names[i], sd->h_epoch[i], &sd->h_p[i], &sd->h_body_md[i], &sd->h_y[0][i], &sd->h_y[1][i]);
		}
		output.flush();
		output.close();
		printf(" done\n");
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
}

void get_number_of_bodies(string& path, ttt_t t, data_representation_t repres)
{
	ns = ngp = nrp = npp = nspl = npl = ntp = 0;
	n_total = 0;

	ttt_t epoch[1];
	body_metadata_t bmd[1];
	param_t p[1];
	vec_t r[1], v[1];

	ifstream input;
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str());
		if (input) 
		{
			cout << "Searching for the records in the input file for epoch: " << number_to_string(t) << " [d] ..." << endl;
			do
			{
				load_body_record(input, 0, epoch, bmd, p, r, v);
			} while (!input.eof() && t > epoch[0]);
			if (input.eof())
			{
				input.close();
				throw string("The end of file was reached and the requested epoch was not found.");
			}
			t = epoch[0];
			cout << "The records in the input file for epoch: " << number_to_string(t) << " [d] was found." << endl;
			printf("Reading and counting the records ");
			do
			{
				g_epoch.push_back(epoch[0]);
				g_bmd.push_back(bmd[0]);
				g_param.push_back(p[0]);
				g_coor.push_back(r[0]);
				g_velo.push_back(v[0]);

				if (n_total % 10 == 0)
				{
					printf(".");
				}
				n_total++;
				switch (bmd[0].body_type)
				{
					case BODY_TYPE_STAR:
						ns++;
						break;
					case BODY_TYPE_GIANTPLANET:
						ngp++;
						break;
					case BODY_TYPE_ROCKYPLANET:
						nrp++;
						break;
					case BODY_TYPE_PROTOPLANET:
						npp++;
						break;
					case BODY_TYPE_SUPERPLANETESIMAL:
						nspl++;
						break;
					case BODY_TYPE_PLANETESIMAL:
						npl++;
						break;
					case BODY_TYPE_TESTPARTICLE:
						ntp++;
						break;
					default:
						throw string("Unknown body type " + number_to_string((int)bmd[0].body_type) + ".");
				}
				load_body_record(input, 0, epoch, bmd, p, r, v);
			} while (t == epoch[0]);
			printf(" done\n");
			if (n_total != g_bmd.size())
			{
				throw string("The n_total does not equal to the size of the vector g_bmd.");
			}

			int n = ns; printf("\t%4d star%s found\n",             n, (n > 1 ? "s were" : " was"));
			n = ngp; printf("\t%4d giant planet%s found\n",        n, (n > 1 ? "s were" : " was"));
			n = nrp; printf("\t%4d rocky planet%s found\n",        n, (n > 1 ? "s were" : " was"));
			n = npp; printf("\t%4d protoplanet%s found\n",         n, (n > 1 ? "s were" : " was"));
			n = nspl; printf("\t%4d super-planetesimal%s found\n", n, (n > 1 ? "s were" : " was"));
			n = npl; printf("\t%4d planetesimal%s found\n",        n, (n > 1 ? "s were" : " was"));
			n = ntp; printf("\t%4d test particle%s found\n",       n, (n > 1 ? "s were" : " was"));
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		input.open(path.c_str(), ios::in | ios::binary);
		if (input) 
		{
			throw string("Binary format reader is not implemeted.");
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	}
	input.close();

    return;
}

void load_binary(ifstream& input, sim_data_t *sim_data)
{
	for (unsigned int type = 0; type < BODY_TYPE_N; type++)
	{
		unsigned int tmp = 0;
		input.read((char*)&tmp, sizeof(tmp));
	}

	char name_buffer[30];
	vec_t* r = sim_data->h_y[0];
	vec_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->h_body_md;
	ttt_t* epoch = sim_data->h_epoch;

	for (unsigned int i = 0; i < n_total; i++)
	{
		memset(name_buffer, 0, sizeof(name_buffer));

		input.read((char*)&epoch[i],  1*sizeof(ttt_t));
		input.read(name_buffer,      30*sizeof(char));
		input.read((char*)&bmd[i],    1*sizeof(body_metadata_t));
		input.read((char*)&p[i],      1*sizeof(param_t));
		input.read((char*)&r[i],      1*sizeof(vec_t));
		input.read((char*)&v[i],      1*sizeof(vec_t));

		body_names.push_back(name_buffer);
	}
}

void load_body_record(ifstream& input, int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v)
{
	int_t	type = 0;
	string	dummy;

	// id
	input >> body_md[k].id;
	// name
	input >> dummy;
	// The names must be less than or equal to 30 chars
	if (dummy.length() > 30)
	{
		dummy = dummy.substr(0, 30);
	}
	body_names.push_back(dummy);
	// body type
	input >> type;
	body_md[k].body_type = static_cast<body_type_t>(type);
	// epoch
	input >> epoch[k];

	// mass, radius density and stokes coefficient
	input >> p[k].mass >> p[k].radius >> p[k].density >> p[k].cd;

	// migration type
	input >> type;
	body_md[k].mig_type = static_cast<migration_type_t>(type);
	// migration stop at
	input >> body_md[k].mig_stop_at;

	// position
	input >> r[k].x >> r[k].y >> r[k].z;
	// velocity
	input >> v[k].x >> v[k].y >> v[k].z;

	r[k].w = v[k].w = 0.0;
}

void load_ascii(ifstream& input, sim_data_t *sim_data)
{
	int ns, ngp, nrp, npp, nspl, npl, ntp;
	input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;

	vec_t* r = sim_data->h_y[0];
	vec_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->h_body_md;
	ttt_t* epoch = sim_data->h_epoch;

	int pcd = 1;

	for (unsigned int i = 0; i < n_total; i++)
	{
		load_body_record(input, i, epoch, bmd, p, r, v);
		if (pcd <= (int)((((var_t)i/(var_t)n_total))*100.0))
		{
			printf(".");
			pcd++;
		}
	}
}

void load(string& path, sim_data_t *sim_data, data_representation_t repres)
{
	cout << "Loading " << path << " ";

	ifstream input;
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str());
		if (input) 
		{
			load_ascii(input, sim_data);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		input.open(path.c_str(), ios::in | ios::binary);
		if (input) 
		{
			load_binary(input, sim_data);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	}
	input.close();

	cout << " done" << endl;
}

int parse_options(int argc, const char **argv, string &iDir, string &input_file, ttt_t &t)
{
	int i = 1;

	while (i < argc)
	{
		string p = argv[i];

		if (     p == "-iDir")
		{
			i++;
			iDir = argv[i];
		}
		else if (p == "-i")
		{
			i++;
			input_file = argv[i];
		}
		else if (p == "-snapshot")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			t = atof(argv[i]);
		}
		else
		{
			throw string("Invalid switch on command-line.");
		}
		i++;
	}

	return 0;
}

int main(int argc, const char **argv)
{
	string iDir;
	string input_file;
	ttt_t t = 0.0;

	sim_data_t* sim_data = 0x0;
	try
	{
		parse_options(argc, argv, iDir, input_file, t);

		string path = file::combine_path(iDir, input_file);
		get_number_of_bodies(path, t, DATA_REPRESENTATION_ASCII);

		sim_data = new sim_data_t;
		sim_data->h_y.resize(2);

		sim_data->h_epoch = g_epoch.data();
		sim_data->h_body_md = g_bmd.data();
		sim_data->h_p = g_param.data();
		sim_data->h_y[0] = g_coor.data();
		sim_data->h_y[1] = g_velo.data();

		{
			ostringstream ss;
			ss.setf(ios::right);
			ss.setf(ios::scientific);

			ss << setw(23) << setprecision(16) << t;

			string output_file = "snapshot_t_" + ss.str();
			path = file::combine_path(iDir, output_file) + ".txt";
			print(path, sim_data);
		}

		{
			for (int i = 1; i < n_total; i++)
			{
				var_t mu = K2 *(sim_data->h_p[0] + sim_data->h_p[i].mass);
			}
		}
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
		return (EXIT_FAILURE);
	}
	delete sim_data;

	return (EXIT_SUCCESS);
}
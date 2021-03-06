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
vector<pp_disk_t::body_metadata_t> g_bmd;
vector<pp_disk_t::param_t> g_param;
vector<var4_t> g_coor;
vector<var4_t> g_velo;

uint32_t ns, ngp, nrp, npp, nspl, npl, ntp, n_total;


#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)

inline string epoch_to_string(ttt_t epoch)
{
	ostringstream ss;
	ss.setf(ios::right);
	ss.setf(ios::scientific);

	ss << setw(23) << setprecision(16) << epoch;

	return ss.str();
}
#else  /* presume POSIX */

inline string epoch_to_string(ttt_t epoch)
{
	ostringstream ss;
	ss.setf(ios::right);
	ss.setf(ios::scientific);

	ss << setw(22) << setprecision(16) << epoch;

	return ss.str();
}
#endif 

void extract_body_record(string& line, int k, ttt_t* epoch, pp_disk_t::body_metadata_t* body_md, pp_disk_t::param_t* p, var4_t* r, var4_t* v);

void print(string &path, pp_disk_t::sim_data_t *sd)
{
	printf("Writing %s to disk .", path.c_str());

	ofstream output(path.c_str(), ios_base::out);
	if (output)
	{
		int_t nBodies = n_total;

		int pcd = 1;
		for (int i = 0; i < nBodies; i++)
		{
			if (pcd <= (int)((((var_t)i/(var_t)nBodies))*100.0))
			{
				printf(".");
				pcd++;
			}
			file::print_body_record_ascii_RED(output, body_names[i], &sd->h_p[i], &sd->h_body_md[i], &sd->h_y[0][i], &sd->h_y[1][i]);
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

void read_snapshot(string& path, ttt_t t, data_rep_t repres)
{
	ttt_t epoch[1];
	pp_disk_t::body_metadata_t bmd[1];
	pp_disk_t::param_t p[1];
	var4_t r[1], v[1];

	ns = ngp = nrp = npp = nspl = npl = ntp = 0;
	n_total = 0;

	ifstream input;
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str());
		if (input) 
		{
			cout << "Searching for the records in the input file with epoch: " << number_to_string(t) << " [d] ..." << endl;
			ttt_t t_tmp = 0.0;
			string line;
			do
			{
				getline(input, line);
				string epoch_str = line.substr(46, 23);

				epoch[0] = atof(epoch_str.c_str());
				if (t_tmp != epoch[0])
				{
					t_tmp = epoch[0];
					printf("t = %20.16le\r", epoch[0]);
				}
			} while (!input.eof() && t > epoch[0]);
			if (input.eof())
			{
				input.close();
				throw string("\nThe end of file was reached and the requested epoch was not found.");
			}

			t = epoch[0];
			cout << "\nThe records in the input file for epoch: " << epoch_to_string(t) << " [d] was found." << endl;
			printf("Reading and counting the records ");
			do
			{
				extract_body_record(line, 0, epoch, bmd, p, r, v);
				if (t != epoch[0])
				{
					break;
				}

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
				getline(input, line);
			} while (!input.eof() && t == epoch[0]);
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
	}
	input.close();

    return;
}

void read_body(string& path, int id, int& n_id_record, data_rep_t repres)
{
	ttt_t epoch[1];
	pp_disk_t::body_metadata_t bmd[1];
	pp_disk_t::param_t p[1];
	var4_t r[1], v[1];

	int n_star_record = 0;
	n_id_record = 0;

	n_total = 0;
	ifstream input;
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str());
		if (input) 
		{
			cout << "Searching for records in the input file with id: " << number_to_string(id) << " ..." << endl;
			string line;
			int id_tmp = 0;
			do
			{
				getline(input, line);
				string id_str = line.substr(0, 8);
				id_tmp = atoi(id_str.c_str());
				if (1 == id_tmp || id == id_tmp)
				{
					if (id == id_tmp)
					{
						n_id_record++;
					}

					if (1 == id_tmp)
					{
						n_star_record++;
					}

					extract_body_record(line, 0, epoch, bmd, p, r, v);
					g_epoch.push_back(epoch[0]);
					g_bmd.push_back(bmd[0]);
					g_param.push_back(p[0]);
					g_coor.push_back(r[0]);
					g_velo.push_back(v[0]);

					if (n_id_record % 10 == 0)
					{
						printf(".");
					}
				}
			} while (1 >= (n_star_record - n_id_record) && !input.eof());

			if (0 == n_id_record)
			{
				input.close();
				throw string("\nThe end of file was reached and the requested id was not found.");
			}
			n_total = 2 * n_id_record;

			printf(" done\n");
			printf("\t%4d record%s found\n", n_id_record, (n_id_record > 1 ? "s were" : " was"));
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
	}
	input.close();

    return;
}

void load_binary(ifstream& input, pp_disk_t::sim_data_t *sim_data)
{
	for (uint32_t type = 0; type < BODY_TYPE_N; type++)
	{
		uint32_t tmp = 0;
		input.read((char*)&tmp, sizeof(tmp));
	}

	char name_buffer[30];
	var4_t* r = sim_data->h_y[0];
	var4_t* v = sim_data->h_y[1];
	pp_disk_t::param_t* p = sim_data->h_p;
	pp_disk_t::body_metadata_t* bmd = sim_data->h_body_md;
	ttt_t* epoch = sim_data->h_epoch;

	for (uint32_t i = 0; i < n_total; i++)
	{
		memset(name_buffer, 0, sizeof(name_buffer));

		input.read((char*)&epoch[i],  1*sizeof(ttt_t));
		input.read(name_buffer,      30*sizeof(char));
		input.read((char*)&bmd[i],    1*sizeof(pp_disk_t::body_metadata_t));
		input.read((char*)&p[i],      1*sizeof(pp_disk_t::param_t));
		input.read((char*)&r[i],      1*sizeof(var4_t));
		input.read((char*)&v[i],      1*sizeof(var4_t));

		body_names.push_back(name_buffer);
	}
}

void extract_body_record(string& line, int k, ttt_t* epoch, pp_disk_t::body_metadata_t* body_md, pp_disk_t::param_t* p, var4_t* r, var4_t* v)
{
	int_t	type = 0;
	string	dummy;

	stringstream ss;
	ss << line;

	// id
	ss >> body_md[k].id;

	// name
	ss >> dummy;
	// The names must be less than or equal to 30 chars
	if (dummy.length() > 30)
	{
		dummy = dummy.substr(0, 30);
	}
	body_names.push_back(dummy);

	// body type
	ss >> type;
	body_md[k].body_type = static_cast<body_type_t>(type);

	// epoch
	ss >> epoch[k];

	// mass, radius density and stokes coefficient
	ss >> p[k].mass >> p[k].radius >> p[k].density >> p[k].cd;

	// migration type
	ss >> type;
	body_md[k].mig_type = static_cast<migration_type_t>(type);

	// migration stop at
	ss >> body_md[k].mig_stop_at;

	// position
	ss >> r[k].x >> r[k].y >> r[k].z;

	// velocity
	ss >> v[k].x >> v[k].y >> v[k].z;

	r[k].w = v[k].w = 0.0;
}

int parse_options(int argc, const char **argv, string &iDir, string &oDir, string &input_file, ttt_t &t, int& id)
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
		else if (p == "-oDir")
		{
			i++;
			oDir = argv[i];
		}
		else if (p == "-i")
		{
			i++;
			input_file = argv[i];
		}
		else if (p == "-snapshot" || p == "-ts")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			t = atof(argv[i]);
		}
		else if (p == "-id")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			id = atoi(argv[i]);
		}
		else
		{
			throw string("Invalid switch on command-line: " + p + ".");
		}
		i++;
	}

	return 0;
}

//Release: -ts 361597.50 -iDir C:\Work\red.cuda.Results\CollisionStatistics\2D\CDM_Step\Aref_1\Run_01 -oDir C:\Work\red.cuda.Results\CollisionStatistics\2D\CDM_Step\Aref_1\Run_01\Snapshot -i result.txt
int main(int argc, const char **argv)
{
	string iDir;
	string oDir;
	string input_file;
	string output_file;
	ttt_t t_snapshot = 0.0;
	int id = 0;
	int n_id_record = 0;

	pp_disk_t::sim_data_t* sim_data = 0x0;
	try
	{
		parse_options(argc, argv, iDir, oDir,input_file, t_snapshot, id);

		string path = file::combine_path(iDir, input_file);
		if (0 < t_snapshot)
		{
			read_snapshot(path, t_snapshot, DATA_REPRESENTATION_ASCII);
		}
		if (0 < id)
		{
			read_body(path, id, n_id_record, DATA_REPRESENTATION_ASCII);
		}

		sim_data = new pp_disk_t::sim_data_t;
		sim_data->h_y.resize(2);
		ALLOCATE_HOST_VECTOR((void **)&(sim_data->h_oe), n_total*sizeof(orbelem_t));

		sim_data->h_epoch   = g_epoch.data();
		sim_data->h_body_md = g_bmd.data();
		sim_data->h_p       = g_param.data();
		sim_data->h_y[0]    = g_coor.data();
		sim_data->h_y[1]    = g_velo.data();

		// create output filename
		if (0 < t_snapshot)
		{
			output_file = "snapshot_t_" + epoch_to_string(sim_data->h_epoch[0]);
		}
		if (0 < id)
		{
			output_file = "body_id_" + redutilcu::number_to_string(id, 6, true);
		}

		if (0 < t_snapshot)
		{
			path = file::combine_path(oDir, output_file) + ".txt";
			print(path, sim_data);
		}		

		{
			if (0 < t_snapshot)
			{
				for (uint32_t i = 1; i < n_total; i++)
				{
					var_t mu = K2 *(sim_data->h_p[0].mass + sim_data->h_p[i].mass);
					var4_t rVec = {sim_data->h_y[0][i].x - sim_data->h_y[0][0].x, sim_data->h_y[0][i].y - sim_data->h_y[0][0].y, sim_data->h_y[0][i].z - sim_data->h_y[0][0].z, 0.0};
					var4_t vVec = {sim_data->h_y[1][i].x - sim_data->h_y[1][0].x, sim_data->h_y[1][i].y - sim_data->h_y[1][0].y, sim_data->h_y[1][i].z - sim_data->h_y[1][0].z, 0.0};
					tools::calc_oe(mu, &rVec, &vVec, (&sim_data->h_oe[i]));
				}
			}

			if (0 < id)
			{
				int j = 0;
				for (uint32_t i = 0; i < n_total; i += 2, j++)
				{
					var_t mu = K2 *(sim_data->h_p[i].mass + sim_data->h_p[i+1].mass);
					var4_t rVec = {sim_data->h_y[0][i+1].x - sim_data->h_y[0][i].x, sim_data->h_y[0][i+1].y - sim_data->h_y[0][i].y, sim_data->h_y[0][i+1].z - sim_data->h_y[0][i].z, 0.0};
					var4_t vVec = {sim_data->h_y[1][i+1].x - sim_data->h_y[1][i].x, sim_data->h_y[1][i+1].y - sim_data->h_y[1][i].y, sim_data->h_y[1][i+1].z - sim_data->h_y[1][i].z, 0.0};
					tools::calc_oe(mu, &rVec, &vVec, (&sim_data->h_oe[j]));
				}
			}

			path = file::combine_path(oDir, output_file) + ".oe.txt";
			ofstream output(path.c_str(), ios_base::out);
			if (output)
			{
				uint32_t n_oe_record = 0;
				if (0 < t_snapshot)
				{
					n_oe_record = n_total;
				}
				if (0 < id)
				{
					n_oe_record = n_id_record;
				}

				for (uint32_t i = 0; i < n_oe_record; i++)
				{
					file::print_oe_record(output, sim_data->h_epoch[2*i], &sim_data->h_oe[i], &sim_data->h_p[2*i+1], &sim_data->h_body_md[2*i+1]);
				}
				output.close();
			}
			else
			{
				throw string("Cannot open " + path + "!");
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

// includes system
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>

// includes project
#include "file_util.h"
#include "tools.h"
#include "util.h"

#include "red_constants.h"
#include "red_macro.h"
#include "red_type.h"

using namespace std;

namespace redutilcu
{
namespace file
{
string combine_path(const string& dir, const string& filename)
{
	if (0 < dir.size())
	{
		if (*(dir.end() - 1) != '/' && *(dir.end() - 1) != '\\')
		{
			return dir + '/' + filename;
		}
		else
		{
			return dir + filename;
		}
	}
	else
	{
		return filename;
	}
}

string get_filename(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of("/\\");
		result = path.substr(pos + 1);
	}

	return result;
}

string get_filename_without_ext(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of("/\\");
		result = path.substr(pos + 1);
		pos = result.find_last_of('.');
		result = result.substr(0, pos);
	}

	return result;
}

string get_directory(const string& path)
{
	string result;

	if (0 < path.size())
	{
		size_t pos = path.find_last_of("/\\");
		// If path does not contain / or \ than path does not contain any directory so return an empty string
		if (pos == string::npos)
		{
			return result;
		}
		// Copy the directory part of path into result
		result = path.substr(0, pos);
	}

	return result;
}

string get_extension(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of('.');
		result = path.substr(pos + 1);
	}

	return result;
}

uint32_t load_ascii_file(const string& path, string& result)
{
	uint32_t n_line = 0;

	std::ifstream file(path.c_str(), ifstream::in);
	if (file)
	{
		string str;
		while (getline(file, str))
		{
			// delete everything after the comment '#' character and the '#'
			str = tools::trim_comment(str);
			str = tools::trim(str);
			if (0 == str.length())
			{
				continue;
			}
			result += str;
			result.push_back('\n');
			n_line++;
		} 	
		file.close();
	}
	else
	{
		throw string("The file '" + path + "' could not opened!\r\n");
	}

	return n_line;
}

void load_binary_file(const string& path, size_t n_data, var_t* data)
{
	ifstream file(path.c_str(), ios::in | ios::binary);
	if (file)
	{
		file.seekg(0, file.end);     //N is the size of file in byte
		size_t N = file.tellg();              
		file.seekg(0, file.beg);
		size_t size = n_data * sizeof(var_t);
		if (size != N)
		{
			throw string("The file '" + path + "' has different number of data than expected!\r\n");
		}
		file.read(reinterpret_cast<char*>(data), size);
		file.close();
	}
	else
	{
		throw string("The file '" + path + "' could not opened!\r\n");
	}
	file.close();
}

void Emese_data_format_to_red_cuda_format(const string& input_path, const string& output_path)
{
	ifstream input(  input_path.c_str(), ios::in | ios::binary);
	ofstream output(output_path.c_str(), ios_base::out);

	if (!input)
	{
		throw string("The file '" + input_path + "' could not opened!\r\n");
	}
	if (!output)
	{
		throw string("The file '" + output_path + "' could not opened!\r\n");
	}

	output << "1 0 0 5000 0 0 5000" << endl;
	if (input && output) 
	{
		ttt_t time = 0.0;        
		int64_t nbodyfilein;
		int64_t lengthChar;      
		char buffer[64];
		var_t id = 0;
		string name;
		string reference;
		var_t x = 0;
		var_t y = 0;
		var_t z = 0;
		var_t vx = 0;
		var_t vy = 0;
		var_t vz = 0;
		var_t m = 0;
		var_t rad = 0;

		input.read(reinterpret_cast<char *>(&time), sizeof(time));
		input.read(reinterpret_cast<char *>(&nbodyfilein), sizeof(nbodyfilein));
		for (int i = 0; i < nbodyfilein; i++)
		{
			input.read(reinterpret_cast<char *>(&id), sizeof(id));

			lengthChar = 0;
			input.read(reinterpret_cast<char *>(&lengthChar), sizeof(lengthChar));            
			input.read(buffer, lengthChar);
			buffer[lengthChar] = 0;
			name = buffer;
			replace(name.begin(), name.end(), ' ', '_'); // replace all ' ' to '_'

			lengthChar = 0;
			input.read(reinterpret_cast<char *>(&lengthChar), sizeof(lengthChar));
			input.read(buffer, lengthChar);
			buffer[lengthChar] = 0;
			reference = buffer; 

			input.read(reinterpret_cast<char *>(&x),  sizeof( x));
			input.read(reinterpret_cast<char *>(&y),  sizeof( y));
			input.read(reinterpret_cast<char *>(&z),  sizeof( z));
			input.read(reinterpret_cast<char *>(&vx), sizeof(vx));
			input.read(reinterpret_cast<char *>(&vy), sizeof(vy));
			input.read(reinterpret_cast<char *>(&vz), sizeof(vz));
			input.read(reinterpret_cast<char *>(&m),  sizeof( m));
			input.read(reinterpret_cast<char *>(&rad),sizeof(rad));

			var4_t	rVec = {x, y, z, 0.0};
			var4_t	vVec = {vx, vy, vz, 0.0};

			pp_disk_t::param_t	        param;
            pp_disk_t::body_metadata_t body_md;

			// red.cuda: id starts from 1
			body_md.id = (int)++id;
			if (1 == body_md.id)
			{
				body_md.body_type = BODY_TYPE_STAR;
			}
			if (1 < body_md.id)
			{
				if (0.0 < m)
				{
					body_md.body_type = BODY_TYPE_PROTOPLANET;
				}
				else
				{
					body_md.body_type = BODY_TYPE_TESTPARTICLE;
				}
			}

			param.cd = 0.0;
			param.mass = m;
			param.radius = rad;
			param.density = tools::calc_density(m, rad);
			body_md.mig_stop_at = 0.0;
			body_md.mig_type = MIGRATION_TYPE_NO;

			file::print_body_record_ascii_RED(output, name, &param, &body_md, &rVec, &vVec);
		}
		input.close();
		output.close();
	}
	else
	{
		throw string("Error reading or writing in Emese_data_format_to_red_cuda_format() function!\r\n");
	}
}

void log_start(ostream& sout, int argc, const char** argv, const char** env, string params)
{
	sout << tools::get_time_stamp(false) << " starting " << argv[0] << endl;
	sout << "Command line arguments: " << endl;
	for (int i = 1; i < argc; i++)
	{
		sout << argv[i] << SEP;
	}
	sout << endl << endl;

	while (*env)
	{
		string s = *env;
#ifdef __GNUC__
		// TODO
		if(      s.find("HOSTNAME=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("USER=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("OSTYPE=") < s.length())
		{
			sout << s << endl;
		}
#else
		if(      s.find("COMPUTERNAME=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("USERNAME=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("OS=") < s.length())
		{
			sout << s << endl;
		}
#endif
		env++;
	}
	sout << endl;

	sout << "Parameters:" << endl << params << endl;
}

void log_start(ostream& sout, int argc, const char** argv, const char** env, string params, bool print_to_screen)
{
	log_start(sout, argc, argv, env, params);
	if (print_to_screen)
	{
		log_start(std::cout, argc, argv, env, params);
	}
}

void log_message(ostream& sout, string msg, bool print_to_screen)
{
	sout << tools::get_time_stamp(false) << SEP << msg << endl;
	if (print_to_screen && sout != cout)
	{
		std::cout << tools::get_time_stamp(false) << SEP << msg << endl;
	}
}

void print_data_info_record_ascii_RED(ofstream& sout, ttt_t t, ttt_t dt, uint32_t dt_CPU, n_objects_t* n_bodies)
{
	static uint32_t int_t_w =  8;
	static uint32_t var_t_w = 25;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	sout << setw(var_t_w) << t << SEP
		 << setw(var_t_w) << dt << SEP
		 << setw(     10) << dt_CPU << SEP;
	for (uint32_t type = 0; type < BODY_TYPE_N; type++)
	{
		sout << setw(int_t_w) << n_bodies->get_n_active_by((body_type_t)type) << SEP;
	}
}

void print_data_info_record_binary_RED(ofstream& sout, ttt_t t, ttt_t dt, uint32_t dt_CPU, n_objects_t* n_bodies)
{
	sout.write((char*)&(t),      sizeof(ttt_t));
	sout.write((char*)&(dt),     sizeof(ttt_t));
	sout.write((char*)&(dt_CPU), sizeof(uint32_t));
	for (uint32_t type = 0; type < BODY_TYPE_N; type++)
	{
		uint32_t n = n_bodies->get_n_active_by((body_type_t)type);
		sout.write((char*)&n, sizeof(n));
	}
}

void print_body_record_ascii_RED(ofstream &sout, string name, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd, var4_t *r, var4_t *v)
{
	static int int_t_w  =  8;
	static int var_t_w  = 25;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	sout << setw(     30) << name             << SEP
		 << setw(int_t_w) << bmd->id          << SEP
		 << setw(      2) << bmd->body_type   << SEP 
		 << setw(      2) << bmd->mig_type    << SEP
		 << setw(var_t_w) << bmd->mig_stop_at << SEP
		 << setw(var_t_w) << p->mass          << SEP
		 << setw(var_t_w) << p->radius        << SEP
		 << setw(var_t_w) << p->density       << SEP
		 << setw(var_t_w) << p->cd            << SEP
		 << setw(var_t_w) << r->x             << SEP
		 << setw(var_t_w) << r->y             << SEP
		 << setw(var_t_w) << r->z             << SEP
		 << setw(var_t_w) << v->x             << SEP
		 << setw(var_t_w) << v->y             << SEP
		 << setw(var_t_w) << v->z             << endl;

    sout.flush();
}

void print_body_record_binary_RED(ofstream &sout, string name, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd, var4_t *r, var4_t *v)
{
	sout.write((char*)name.c_str(), 30*sizeof(char));
	sout.write((char*)bmd, sizeof(pp_disk_t::body_metadata_t));
	sout.write((char*)p,   sizeof(pp_disk_t::param_t));
	sout.write((char*)r,   3*sizeof(var_t));
	sout.write((char*)v,   3*sizeof(var_t));
}

void print_body_record_HIPERION(ofstream &sout, string name, var_t epoch, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *body_md, var4_t *r, var4_t *v)
{
	static int ids[4] = {0, 10, 20, 10000000};

	static int int_t_w  =  8;
	static int var_t_w  = 25;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	int id = 0;
	// NOTE: this is not the beta parameter but derived from it
	var_t _beta = 0.0;
	switch (body_md->body_type)
	{
	case BODY_TYPE_STAR:
		id = ids[0];
		ids[0]++;
		_beta = 0.0;
		break;
	case BODY_TYPE_GIANTPLANET:
		id = ids[1];
		ids[1]++;
		_beta = 0.0;
		break;
	case BODY_TYPE_ROCKYPLANET:
		id = ids[1];
		ids[1]++;
		_beta = 0.0;
		break;
	case BODY_TYPE_PROTOPLANET:
		id = ids[2];
		ids[2]++;
		_beta = p->density;
		break;
	case BODY_TYPE_SUPERPLANETESIMAL:
		break;
	case BODY_TYPE_PLANETESIMAL:
		id = ids[2];
		ids[2]++;
		_beta = p->density;
		break;
	case BODY_TYPE_TESTPARTICLE:
		id = ids[3];
		ids[3]++;
		_beta = 1.0;
		break;
	default:
		throw string("Parameter 'body_type' is out of range.");
	}

	var_t eps = 0.0;

	sout << setw(int_t_w) << id      << SEP
		 << setw(var_t_w) << p->mass << SEP
		 << setw(var_t_w) << r->x    << SEP
		 << setw(var_t_w) << r->y    << SEP
		 << setw(var_t_w) << r->z    << SEP
		 << setw(var_t_w) << v->x / constants::Gauss << SEP
		 << setw(var_t_w) << v->y / constants::Gauss << SEP
		 << setw(var_t_w) << v->z / constants::Gauss << SEP
		 << setw(var_t_w) << eps     << SEP
		 << setw(var_t_w) << _beta   << endl;

    sout.flush();
}

void print_body_record_Emese(ofstream &sout, string name, var_t epoch, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *body_md, var4_t *r, var4_t *v)
{
	const char* body_type_name[] = 
	{
		"STAR",
		"GIANTPLANET",
		"ROCKYPLANET",
		"PROTOPLANET",
		"SUPERPLANETESIMAL",
		"PLANETESIMAL",
		"TESTPARTICLE",
	};

	static int int_t_w  = 25;
	static int var_t_w  = 25;

	sout.precision(16);
	sout.setf(ios::left);
	sout.setf(ios::scientific);

	// NOTE: Emese start the ids from 0, red starts from 1.
	sout << setw(int_t_w) << noshowpos << body_md->id - 1
		 << setw(     25) << name
		 << setw(     25) << noshowpos << body_type_name[body_md->body_type]
		 << setw(var_t_w) << showpos << r->x
		 << setw(var_t_w) << showpos << r->y
		 << setw(var_t_w) << showpos << r->z
		 << setw(var_t_w) << showpos << v->x
		 << setw(var_t_w) << showpos << v->y
		 << setw(var_t_w) << showpos << v->z
		 << setw(var_t_w) << showpos << p->mass
		 << setw(var_t_w) << showpos << p->radius << endl;

    sout.flush();
}

void print_oe_record(ofstream &sout, orbelem_t* oe)
{
	static int var_t_w  = 15;

	sout.precision(6);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	sout << setw(var_t_w) << oe->sma << SEP 
         << setw(var_t_w) << oe->ecc << SEP 
         << setw(var_t_w) << oe->inc << SEP 
         << setw(var_t_w) << oe->peri << SEP 
         << setw(var_t_w) << oe->node << SEP 
         << setw(var_t_w) << oe->mean << endl;

	sout.flush();
}

void print_oe_record(ofstream &sout, orbelem_t* oe, pp_disk_t::param_t *p)
{
	static int var_t_w  = 15;

	sout.precision(6);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	sout << setw(var_t_w) << oe->sma    << SEP 
         << setw(var_t_w) << oe->ecc    << SEP 
         << setw(var_t_w) << oe->inc    << SEP 
         << setw(var_t_w) << oe->peri   << SEP 
         << setw(var_t_w) << oe->node   << SEP 
         << setw(var_t_w) << oe->mean   << SEP
         << setw(var_t_w) << p->mass    << SEP
         << setw(var_t_w) << p->radius  << SEP
         << setw(var_t_w) << p->density << SEP
         << setw(var_t_w) << p->cd      << endl;

	sout.flush();
}

void print_oe_record(ofstream &sout, ttt_t epoch, orbelem_t* oe, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd)
{
	static int var_t_w  = 15;
	static int int_t_w  = 7;

	sout.precision(6);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	sout << setw(var_t_w) << epoch          << SEP 
		 << setw(int_t_w) << bmd->id        << SEP 
		 << setw(      2) << bmd->body_type << SEP 
         << setw(var_t_w) << p->mass        << SEP
         << setw(var_t_w) << p->radius      << SEP
         << setw(var_t_w) << p->density     << SEP
         << setw(var_t_w) << p->cd          << SEP
		 << setw(var_t_w) << oe->sma        << SEP 
         << setw(var_t_w) << oe->ecc        << SEP 
         << setw(var_t_w) << oe->inc        << SEP 
         << setw(var_t_w) << oe->peri       << SEP 
         << setw(var_t_w) << oe->node       << SEP 
         << setw(var_t_w) << oe->mean       << endl;

	sout.flush();
}

void load_data_info_record_ascii( ifstream& input, var_t& t, var_t& dt, uint32_t& dt_CPU, n_objects_t** n_bodies)
{
	uint32_t ns, ngp, nrp, npp, nspl, npl, ntp;
	ns = ngp = nrp = npp = nspl = npl = ntp = 0;

	input >> t >> dt >> dt_CPU;
	input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;

	*n_bodies = new n_objects_t(ns, ngp, nrp, npp, nspl, npl, ntp);
}

void load_data_info_record_binary(ifstream& input, var_t& t, var_t& dt, uint32_t& dt_CPU, n_objects_t** n_bodies)
{
	uint32_t ns, ngp, nrp, npp, nspl, npl, ntp;
	ns = ngp = nrp = npp = nspl = npl = ntp = 0;

	input.read((char*)&t,      sizeof(ttt_t));
	input.read((char*)&dt,     sizeof(ttt_t));
	input.read((char*)&dt_CPU, sizeof(uint32_t));

	input.read((char*)&ns,   sizeof(uint32_t));
	input.read((char*)&ngp,  sizeof(uint32_t));
	input.read((char*)&nrp,  sizeof(uint32_t));
	input.read((char*)&npp,  sizeof(uint32_t));
	input.read((char*)&nspl, sizeof(uint32_t));
	input.read((char*)&npl,  sizeof(uint32_t));
	input.read((char*)&ntp,  sizeof(uint32_t));

	*n_bodies = new n_objects_t(ns, ngp, nrp, npp, nspl, npl, ntp);
}

void load_data_record_ascii(ifstream& input, std::string& name, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd, var4_t *r, var4_t *v)
{
	int_t	type = 0;
	string	buffer;

	// 1. field: name
	input >> buffer;
	// The names must be less than or equal to 30 chars
	if (buffer.length() > 30)
	{
		buffer = buffer.substr(0, 30);
	}
	name = buffer;

	// 2. field: id
	input >> bmd->id;
	// 3. field: body type
	input >> type;
	bmd->body_type = static_cast<body_type_t>(type);
	// 4. field:  migration type
	input >> type;
	bmd->mig_type = static_cast<migration_type_t>(type);
	// 5. field: migration stop at
	input >> bmd->mig_stop_at;

	// 6. field: mass
	// 7. field: radius
	// 8. field: density
	// 9. field: stokes coefficient
	input >> p->mass >> p->radius >> p->density >> p->cd;

	// 10. field: x
	// 11. field: y
	// 12. field: z
	input >> r->x >> r->y >> r->z;
	// 13. field: vx
	// 14. field: vy
	// 15. field: vz
	input >> v->x >> v->y >> v->z;
	r->w = v->w = 0.0;
}

void load_data_record_binary(ifstream& input, std::string& name, pp_disk_t::param_t *p, pp_disk_t::body_metadata_t *bmd, var4_t *r, var4_t *v)
{
	char buffer[30];
	memset(buffer, 0, sizeof(buffer));

	input.read(buffer,      30*sizeof(char));
	input.read((char*)bmd,  1*sizeof(pp_disk_t::body_metadata_t));
	input.read((char*)p,    1*sizeof(pp_disk_t::param_t));
	input.read((char*)r,    3*sizeof(var_t));
	input.read((char*)v,    3*sizeof(var_t));

	name = buffer;
}

} /* file */
} /* redutilcu */

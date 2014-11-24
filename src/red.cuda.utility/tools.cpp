// includes system
#include <iostream>
#include <sstream>
#include <cctype>
#include <ctime>
#include <string>

// includes project
#include "tools.h"
#include "red_macro.h"

using namespace std;

namespace redutilcu
{
namespace tools
{
bool is_number(const string& str)
{
   for (size_t i = 0; i < str.length(); i++) {
	   if (std::isdigit(str[i]) || str[i] == 'e' || str[i] == 'E' || str[i] == '.' || str[i] == '-' || str[i] == '+')
           continue;
	   else
		   return false;
   }
   return true;
}

/// Removes all trailing white-space characters from the current std::string object.
void trim_right(string& str)
{
	// trim trailing spaces
	size_t endpos = str.find_last_not_of(" \t");
	if (string::npos != endpos ) {
		str = str.substr( 0, endpos+1 );
	}
}

/// Removes all trailing characters after the first # character
void trim_right(string& str, char c)
{
	// trim trailing spaces

	size_t endpos = str.find(c);
	if (string::npos != endpos ) {
		str = str.substr( 0, endpos);
	}
}

/// Removes all leading white-space characters from the current std::string object.
void trim_left(string& str)
{
	// trim leading spaces
	size_t startpos = str.find_first_not_of(" \t");
	if (string::npos != startpos ) {
		str = str.substr( startpos );
	}
}

/// Removes all leading and trailing white-space characters from the current std::string object.
void trim(string& str)
{
	trim_right(str);
	trim_left(str);
}

string get_time_stamp()
{
	static char time_stamp[20];
	time_t now = time(0);
	strftime(time_stamp, 20, "%Y-%m-%d %H:%M:%S", localtime(&now));

	return string(time_stamp);
}

string convert_time_t(time_t t)
{
	string result;

	ostringstream convert;	// stream used for the conversion
	convert << t;			// insert the textual representation of 't' in the characters in the stream
	result = convert.str();

	return result;
}

var_t get_total_mass(int n, body_type_t type, sim_data_t *sim_data)
{
	var_t totalMass = 0.0;

	param_t* p = sim_data->p;
	for (int j = n - 1; j >= 0; j--)
	{
		if (sim_data->body_md[j].body_type == type)
		{
			totalMass += p[j].mass;
		}
	}

	return totalMass;
}

var_t get_total_mass(int n, sim_data_t *sim_data)
{
	var_t totalMass = 0.0;

	param_t* p = sim_data->p;
	for (int j = n - 1; j >= 0; j--)
	{
		totalMass += p[j].mass;
	}

	return totalMass;
}

void compute_bc(int n, sim_data_t *sim_data, vec_t* R0, vec_t* V0)
{
	const param_t* p = sim_data->p;
	const vec_t* r = sim_data->y[0];
	const vec_t* v = sim_data->y[1];

	for (int j = n - 1; j >= 0; j-- )
	{
		var_t m = p[j].mass;
		R0->x += m * r[j].x;
		R0->y += m * r[j].y;
		R0->z += m * r[j].z;

		V0->x += m * v[j].x;
		V0->y += m * v[j].y;
		V0->z += m * v[j].z;
	}
	var_t M0 = get_total_mass(n, sim_data);

	R0->x /= M0;	R0->y /= M0;	R0->z /= M0;
	V0->x /= M0;	V0->y /= M0;	V0->z /= M0;
}

void transform_to_bc(int n, sim_data_t *sim_data)
{
	cout << "Transforming to barycentric system ... ";

	// Position and velocity of the system's barycenter
	vec_t R0 = {0.0, 0.0, 0.0, 0.0};
	vec_t V0 = {0.0, 0.0, 0.0, 0.0};

	compute_bc(n, sim_data, &R0, &V0);

	vec_t* r = sim_data->y[0];
	vec_t* v = sim_data->y[1];
	// Transform the bodies coordinates and velocities
	for (int j = n - 1; j >= 0; j-- )
	{
		r[j].x -= R0.x;		r[j].y -= R0.y;		r[j].z -= R0.z;
		v[j].x -= V0.x;		v[j].y -= V0.y;		v[j].z -= V0.z;
	}

	cout << "done" << endl;
}

var_t calculate_radius(var_t m, var_t density)
{
	static var_t four_pi_over_three = 4.1887902047863909846168578443727;

	return pow(1.0/four_pi_over_three * m/density, 1.0/3.0);
}

var_t calculate_density(var_t m, var_t R)
{
	static var_t four_pi_over_three = 4.1887902047863909846168578443727;

	if (R == 0.0)
	{
		return 0.0;
	}
	return m / (four_pi_over_three * CUBE(R));
}

var_t caclulate_mass(var_t R, var_t density)
{
	static var_t four_pi_over_three = 4.1887902047863909846168578443727;

	return four_pi_over_three * CUBE(R) * density;
}

int	kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E)
{
	if (ecc == 0.0 || mean == 0.0 || mean == PI)
	{
        *E = mean;
		return 0;
    }
    *E = mean + ecc * (sin(mean)) / (1.0 - sin(mean + ecc) + sin(mean));
    var_t E1 = 0.0;
    var_t error;
    int_t step = 0;
    do
	{
        E1 = *E - (*E - ecc * sin(*E) - mean) / (1.0 - ecc * cos(*E));
        error = fabs(E1 - *E);
        *E = E1;
    } while (error > eps && step++ <= 15);
	if (step > 15 )
	{
		return 1;
	}

	return 0;
}

int calculate_phase(var_t mu, const orbelem_t* oe, vec_t* rVec, vec_t* vVec)
{
    var_t ecc = oe->ecc;
	var_t E = 0.0;
	if (kepler_equation_solver(ecc, oe->mean, 1.0e-14, &E) == 1)
	{
		return 1;
	}
    var_t v = 2.0 * atan(sqrt((1.0 + ecc) / (1.0 - ecc)) * tan(E / 2.0));

    var_t p = oe->sma * (1.0 - SQR(ecc));
    var_t r = p / (1.0 + ecc * cos(v));
    var_t kszi = r * cos(v);
    var_t eta = r * sin(v);
    var_t vKszi = -sqrt(mu / p) * sin(v);
    var_t vEta = sqrt(mu / p) * (ecc + cos(v));

    var_t cw = cos(oe->peri);
    var_t sw = sin(oe->peri);
    var_t cO = cos(oe->node);
    var_t sO = sin(oe->node);
    var_t ci = cos(oe->inc);
    var_t si = sin(oe->inc);

    vec_t P;
	P.x = cw * cO - sw * sO * ci;
	P.y = cw * sO + sw * cO * ci;
	P.z = sw * si;
    vec_t Q;
	Q.x = -sw * cO - cw * sO * ci;
	Q.y = -sw * sO + cw * cO * ci;
	Q.z = cw * si;

	rVec->x = kszi * P.x + eta * Q.x;
	rVec->y = kszi * P.y + eta * Q.y;
	rVec->z = kszi * P.z + eta * Q.z;

	vVec->x = vKszi * P.x + vEta * Q.x;
	vVec->y = vKszi * P.y + vEta * Q.y;
	vVec->z = vKszi * P.z + vEta * Q.z;

	return 0;
}

} /* tools */
} /* redutilcu */

// includes system
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cmath>

// includes project
#include "file_util.h"
#include "tools.h"
#include "util.h"

#include "red_macro.h"
#include "red_constants.h"


namespace redutilcu
{

static sim_data* create_sim_data()
{
	sim_data_t *sim_data = new sim_data_t;

	allocate_host_storage(sim_data, 4);

	// Populate 0. body's x metadata
	sim_data->h_body_md[0].id = 1;
	sim_data->h_body_md[0].body_type = BODY_TYPE_STAR;
	sim_data->h_body_md[0].mig_type = MIGRATION_TYPE_NO;
	sim_data->h_body_md[0].mig_stop_at = 0.0;

	// Populate 1. body's x metadata
	sim_data->h_body_md[1].id = 1;
	sim_data->h_body_md[1].body_type = BODY_TYPE_GIANTPLANET;
	sim_data->h_body_md[1].mig_type = MIGRATION_TYPE_NO;
	sim_data->h_body_md[1].mig_stop_at = 0.0;

	// Populate 2. body's x metadata
	sim_data->h_body_md[2].id = 2;
	sim_data->h_body_md[2].body_type = BODY_TYPE_GIANTPLANET;
	sim_data->h_body_md[2].mig_type = MIGRATION_TYPE_NO;
	sim_data->h_body_md[2].mig_stop_at = 0.0;

	// Populate 3. body's x metadata
	sim_data->h_body_md[3].id = 3;
	sim_data->h_body_md[3].body_type = BODY_TYPE_PROTOPLANET;
	sim_data->h_body_md[3].mig_type = MIGRATION_TYPE_NO;
	sim_data->h_body_md[3].mig_stop_at = 0.0;

	// Populate 0. body's x coordinate
	sim_data->h_y[0][0].x = 0.0;
	sim_data->h_y[0][0].y = 0.0;
	sim_data->h_y[0][0].z = 0.0;
	sim_data->h_y[0][0].w = 0.0;
	
	// Populate 1. body's x coordinate
	sim_data->h_y[0][1].x = 1.0;
	sim_data->h_y[0][1].y = 0.0;
	sim_data->h_y[0][1].z = 0.0;
	sim_data->h_y[0][1].w = 0.0;

	// Populate 2. body's x coordinate
	sim_data->h_y[0][2].x = 1.0;
	sim_data->h_y[0][2].y = 1.0;
	sim_data->h_y[0][2].z = 0.0;
	sim_data->h_y[0][2].w = 0.0;

	// Populate 3. body's x coordinate
	sim_data->h_y[0][3].x = 0.0;
	sim_data->h_y[0][3].y = 1.0;
	sim_data->h_y[0][3].z = 0.0;
	sim_data->h_y[0][3].w = 0.0;

	// Populate 0. body's x velocity
	sim_data->h_y[1][0].x = 1.0;
	sim_data->h_y[1][0].y = 0.0;
	sim_data->h_y[1][0].z = 0.0;
	sim_data->h_y[1][0].w = 0.0;
	
	// Populate 1. body's x velocity
	sim_data->h_y[1][1].x = 2.0;
	sim_data->h_y[1][1].y = 0.0;
	sim_data->h_y[1][1].z = 0.0;
	sim_data->h_y[1][1].w = 0.0;

	// Populate 2. body's x velocity
	sim_data->h_y[1][2].x = 2.0;
	sim_data->h_y[1][2].y = 0.0;
	sim_data->h_y[1][2].z = 0.0;
	sim_data->h_y[1][2].w = 0.0;

	// Populate 3. body's x velocity
	sim_data->h_y[1][3].x = 1.0;
	sim_data->h_y[1][3].y = 0.0;
	sim_data->h_y[1][3].z = 0.0;
	sim_data->h_y[1][3].w = 0.0;

	// Set the mass
	sim_data->h_p[0].mass = 1.0;
	sim_data->h_p[1].mass = 1.0;
	sim_data->h_p[2].mass = 1.0;
	sim_data->h_p[3].mass = 1.0;

	// Set the radius
	sim_data->h_p[0].radius = 1.0;
	sim_data->h_p[1].radius = 1.0;
	sim_data->h_p[2].radius = 1.0;
	sim_data->h_p[3].radius = 1.0;

	// Set the density
	sim_data->h_p[0].density = 0.23873241463784300365332564505877;
	sim_data->h_p[1].density = 0.23873241463784300365332564505877;
	sim_data->h_p[2].density = 0.23873241463784300365332564505877;
	sim_data->h_p[3].density = 0.23873241463784300365332564505877;

	// Set the Stokes-coefficient
	sim_data->h_p[0].cd = 0.0;
	sim_data->h_p[1].cd = 0.0;
	sim_data->h_p[2].cd = 0.0;
	sim_data->h_p[3].cd = 0.0;

	return sim_data;
}

static void test_util()
{
	//__host__ __device__ vec_t rotate_2D_vector(var_t theta, var_t v_r, var_t v_theta);

	//template <typename T>
	//std::string number_to_string(T number);
	//int device_query(ostream& sout, int id_dev);

	//void allocate_host_vector(  void **ptr, size_t size,           const char *file, int line);
	//void allocate_device_vector(void **ptr, size_t size,           const char *file, int line);
	//void allocate_vector(       void **ptr, size_t size, bool cpu, const char *file, int line);

	//#define ALLOCATE_HOST_VECTOR(  ptr, size)      (allocate_host_vector(  ptr, size,      __FILE__, __LINE__))
	//#define ALLOCATE_DEVICE_VECTOR(ptr, size)      (allocate_device_vector(ptr, size,      __FILE__, __LINE__))
	//#define ALLOCATE_VECTOR(       ptr, size, cpu) (allocate_vector(       ptr, size, cpu, __FILE__, __LINE__))

	//void free_host_vector(  void **ptr,           const char *file, int line);
	//void free_device_vector(void **ptr,           const char *file, int line);
	//void free_vector(       void **ptr, bool cpu, const char *file, int line);

	//#define FREE_HOST_VECTOR(  ptr)      (free_host_vector(  ptr,      __FILE__, __LINE__))
	//#define FREE_DEVICE_VECTOR(ptr)      (free_device_vector(ptr,      __FILE__, __LINE__))
	//#define FREE_VECTOR(       ptr, cpu) (free_vector(       ptr, cpu, __FILE__, __LINE__))

	//void allocate_host_storage(sim_data_t *sd, int n);
	//void allocate_device_storage(sim_data_t *sd, int n);

	//void deallocate_host_storage(sim_data_t *sd);
	//void deallocate_device_storage(sim_data_t *sd);

	//void copy_vector_to_device(void* dst, const void *src, size_t count);
	//void copy_vector_to_host(  void* dst, const void *src, size_t count);
	//void copy_vector_d2d(      void* dst, const void *src, size_t count);

	//void copy_constant_to_device(const void* dst, const void *src, size_t count);

	//void set_device(int id_a_dev, bool verbose);
	//void print_array(string path, int n, var_t *data, computing_device_t comp_dev);

	const char test_set[] = "test_util";

	fprintf(stderr, "TEST: %s\n", test_set);

	// Test rotate_2D_vector()
	{
		char test_func[] = "rotate_2D_vector";

		var_t theta = 0.0;
		vec_t v = {0, 1, 0, 0};

		vec_t expected = {0, 1, 0, 0};
		vec_t result = rotate_2D_vector(theta, v);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-16 < fabs(result.x - expected.x) && 1.0e-16 < fabs(result.y - expected.y))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf, %25.16lf\n\t\t But was: %25.16lf, %25.16lf\n", expected.x, expected.y, result.x, result.y);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		theta = PI / 2.0;

		expected.x = -1; expected.y = 0;
		result = rotate_2D_vector(theta, v);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-16 < fabs(result.x - expected.x) && 1.0e-16 < fabs(result.y - expected.y))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf, %25.16lf\n\t\t But was: %25.16lf, %25.16lf\n", expected.x, expected.y, result.x, result.y);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		theta = PI;

		expected.x = 0; expected.y = -1;
		result = rotate_2D_vector(theta, v);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-16 < fabs(result.x - expected.x) && 1.0e-16 < fabs(result.y - expected.y))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf, %25.16lf\n\t\t But was: %25.16lf, %25.16lf\n", expected.x, expected.y, result.x, result.y);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		theta = 3.0 * PI / 2.0;

		expected.x = 1; expected.y = 0;
		result = rotate_2D_vector(theta, v);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-16 < fabs(result.x - expected.x) && 1.0e-16 < fabs(result.y - expected.y))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf, %25.16lf\n\t\t But was: %25.16lf, %25.16lf\n", expected.x, expected.y, result.x, result.y);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		theta = PI / 4.0;

		expected.x = -1/sqrt(2.0); expected.y = 1/sqrt(2.0);
		result = rotate_2D_vector(theta, v);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-16 < fabs(result.x - expected.x) && 1.0e-16 < fabs(result.y - expected.y))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf, %25.16lf\n\t\t But was: %25.16lf, %25.16lf\n", expected.x, expected.y, result.x, result.y);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		theta = 3.0 * PI / 4.0;

		expected.x = -1/sqrt(2.0); expected.y = -1/sqrt(2.0);
		result = rotate_2D_vector(theta, v);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-16 < fabs(result.x - expected.x) && 1.0e-16 < fabs(result.y - expected.y))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf, %25.16lf\n\t\t But was: %25.16lf, %25.16lf\n", expected.x, expected.y, result.x, result.y);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		theta = 5.0 * PI / 4.0;

		expected.x = 1/sqrt(2.0); expected.y = -1/sqrt(2.0);
		result = rotate_2D_vector(theta, v);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-16 < fabs(result.x - expected.x) && 1.0e-16 < fabs(result.y - expected.y))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf, %25.16lf\n\t\t But was: %25.16lf, %25.16lf\n", expected.x, expected.y, result.x, result.y);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		theta = 7.0 * PI / 4.0;

		expected.x = 1/sqrt(2.0); expected.y = 1/sqrt(2.0);
		result = rotate_2D_vector(theta, v);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-16 < fabs(result.x - expected.x) && 1.0e-16 < fabs(result.y - expected.y))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf, %25.16lf\n\t\t But was: %25.16lf, %25.16lf\n", expected.x, expected.y, result.x, result.y);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}
}

static void test_tools()
{
	//bool is_number(const string& str);
	//void rtrim(string& str);
	//void ltrim(string& str);
	//void trim(string& str);
	//string get_time_stamp();
	//string convert_time_t(time_t t);

	////! Computes the total mass of the system
	//var_t get_total_mass(int n, const sim_data_t *sim_data);
	////! Computes the total mass of the bodies with type in the system
	//var_t get_total_mass(int n, body_type_t type, const sim_data_t *sim_data);
	//void calc_bc(int n, bool verbose, const sim_data_t *sim_data, vec_t* R0, vec_t* V0);
	//void transform_to_bc(int n, bool verbose, const sim_data_t *sim_data);

	//var_t calc_radius(var_t m, var_t density);
	//var_t calc_density(var_t m, var_t R);
	//var_t caclulate_mass(var_t R, var_t density);

	//void calc_position_after_collision(var_t m1, var_t m2, const vec_t* r1, const vec_t* r2, vec_t& r);
	//void calc_velocity_after_collision(var_t m1, var_t m2, const vec_t* v1, const vec_t* v2, vec_t& v);
	//void calc_physical_properties(var_t m1, var_t m2, var_t r1, var_t r2, var_t cd, param_t &p);

	//int kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E);
	//int calculate_phase(var_t mu, const orbelem_t* oe, vec_t* rVec, vec_t* vVec);

	//void print_vector(vec_t *v);

	const char test_set[] = "test_tools";

	fprintf(stderr, "TEST: %s\n", test_set);

	// Test is_number()
	{
		char test_func[] = "is_number";

		string arg = "1.0";
		var_t num = 1.0;
		bool expected = true;
		bool result = tools::is_number(arg);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected && num == atof(arg.c_str()))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %5s\n\t\t But was: %5s\n", (expected ? "true" : "false"), (result ? "true" : "false"));
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
		
		arg = "-1.0";
		num = -1.0;
		expected = true;
		result = tools::is_number(arg);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected && num == atof(arg.c_str()))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %5s\n\t\t But was: %5s\n", (expected ? "true" : "false"), (result ? "true" : "false"));
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		arg = "+1.0e3";
		num = 1000.0;
		expected = true;
		result = tools::is_number(arg);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected && num == atof(arg.c_str()))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %5s\n\t\t But was: %5s\n", (expected ? "true" : "false"), (result ? "true" : "false"));
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		arg = "1.e3";
		num = 1000.0;
		expected = true;
		result = tools::is_number(arg);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected && num == atof(arg.c_str()))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %5s\n\t\t But was: %5s\n", (expected ? "true" : "false"), (result ? "true" : "false"));
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		arg = "1.E-3";
		num = 0.001;
		expected = true;
		result = tools::is_number(arg);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected && num == atof(arg.c_str()))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %5s\n\t\t But was: %5s\n", (expected ? "true" : "false"), (result ? "true" : "false"));
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		arg = "1.g-3";
		expected = false;
		result = tools::is_number(arg);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %5s\n\t\t But was: %5s\n", (expected ? "true" : "false"), (result ? "true" : "false"));
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test trim_right()
	{
		char test_func[] = "trim_right";

		string result = "  This string will be trimed to the right   ";
		string expected = "  This string will be trimed to the right";
		result = tools::rtrim(result);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %s\n\t\t But was: %s\n", expected.c_str(), result.c_str());
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
		
		result = "This string will be trimed to the right after the 'x' character  ";
		expected = "This string will be trimed to the right after the '";
		result = tools::rtrim(result, "x");

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %s\n\t\t But was: %s\n", expected.c_str(), result.c_str());
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test trim_left()
	{
		char test_func[] = "trim_left";

		string result = "  This string will be trimed to the left	";
		string expected = "This string will be trimed to the left	";
		result = tools::ltrim(result);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %s\n\t\t But was: %s\n", expected.c_str(), result.c_str());
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		result = "			c  This string will be trimed to the left		";
		expected = "c  This string will be trimed to the left		";
		result = tools::ltrim(result);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %s\n\t\t But was: %s\n", expected.c_str(), result.c_str());
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test trim()
	{
		char test_func[] = "trim";

		string result = "  This string will be trimed to the right as well as to the left		";
		string expected = "This string will be trimed to the right as well as to the left";
		result = tools::trim(result);

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %s\n\t\t But was: %s\n", expected.c_str(), result.c_str());
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_time_stamp()
	{
		char test_func[] = "get_time_stamp";

		string result = tools::get_time_stamp(false);

		fprintf(stderr, "\t%s(): [should print the actual date and time] ", test_func);
		fprintf(stderr, "%s: PASSED ??\n", result.c_str());
	}

	// Test convert_time_t()
	{
		char test_func[] = "convert_time_t";

		string result = tools::convert_time_t((time_t)60);
		string expected = "60";

		fprintf(stderr, "\t%s(): ", test_func);
		if (result != expected)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %s\n\t\t But was: %s\n", expected.c_str(), result.c_str());
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_total_mass()
	{
		char test_func[] = "get_total_mass";
	
		sim_data_t* sim_data = create_sim_data();

		var_t expected = 4.0;
		var_t result = tools::get_total_mass(4, sim_data);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected - result))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		delete sim_data;
	}

	// Test get_total_mass()
	{
		char test_func[] = "get_total_mass";
	
		sim_data_t* sim_data = create_sim_data();

		var_t expected = 1.0;
		var_t result = tools::get_total_mass(4, BODY_TYPE_STAR, sim_data);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected - result))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		expected = 2.0;
		result = tools::get_total_mass(4, BODY_TYPE_GIANTPLANET, sim_data);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected - result))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		expected = 1.0;
		result = tools::get_total_mass(4, BODY_TYPE_PROTOPLANET, sim_data);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected - result))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		expected = 0.0;
		result = tools::get_total_mass(4, BODY_TYPE_ROCKYPLANET, sim_data);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected - result))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		delete sim_data;
	}

	// Test calc_bc()
	{
		char test_func[] = "calc_bc";
	
		sim_data_t* sim_data = create_sim_data();

		vec_t expected_R0 = {0.5, 0.5, 0.0, 0.0};
		vec_t expected_V0 = {1.5, 0.0, 0.0, 0.0};

		vec_t result_R0 = {0.0, 0.0, 0.0, 0.0};
		vec_t result_V0 = {0.0, 0.0, 0.0, 0.0};

		tools::calc_bc(4, false, sim_data, &result_R0, &result_V0);

		var_t dr = fabs(expected_R0.x - result_R0.x) + 
			       fabs(expected_R0.y - result_R0.y) + 
				   fabs(expected_R0.z - result_R0.z) +
				   fabs(expected_R0.w - result_R0.w);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < dr)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", 0.0, dr);
		}
		else
		{
			fprintf(stderr, "R0 PASSED\n");
		}

		var_t dv = fabs(expected_V0.x - result_V0.x) + 
			       fabs(expected_V0.y - result_V0.y) + 
				   fabs(expected_V0.z - result_V0.z) +
				   fabs(expected_V0.w - result_V0.w);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < dv)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", 0.0, dv);
		}
		else
		{
			fprintf(stderr, "V0 PASSED\n");
		}

		delete sim_data;
	}

	// Test transform_to_bc()
	{
		char test_func[] = "transform_to_bc";
	
		sim_data_t* sim_data = create_sim_data();

		tools::transform_to_bc(4, false, sim_data);

		vec_t r0[] = { 
			            {-0.5, -0.5, 0.0, 0.0},
			            { 0.5, -0.5, 0.0, 0.0},
			            { 0.5,  0.5, 0.0, 0.0},
			            {-0.5,  0.5, 0.0, 0.0},
		              };

		for (int i = 0; i < 4; i++)
		{
			var_t dr = fabs(r0[i].x - sim_data->h_y[0][i].x) + 
					   fabs(r0[i].y - sim_data->h_y[0][i].y) + 
					   fabs(r0[i].z - sim_data->h_y[0][i].z) +
					   fabs(r0[i].w - sim_data->h_y[0][i].w);

			fprintf(stderr, "\t%s(): ", test_func);
			if (1.0e-15 < dr)
			{
				fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", 0.0, dr);
			}
			else
			{
				fprintf(stderr, "R PASSED\n");
			}
		}

		vec_t v0[] = { 
			            {-0.5,  0.0, 0.0, 0.0},
			            { 0.5,  0.0, 0.0, 0.0},
			            { 0.5,  0.0, 0.0, 0.0},
			            {-0.5,  0.0, 0.0, 0.0},
		              };

		for (int i = 0; i < 4; i++)
		{
			var_t dv = fabs(v0[i].x - sim_data->h_y[1][i].x) + 
					   fabs(v0[i].y - sim_data->h_y[1][i].y) + 
					   fabs(v0[i].z - sim_data->h_y[1][i].z) +
					   fabs(v0[i].w - sim_data->h_y[1][i].w);

			fprintf(stderr, "\t%s(): ", test_func);
			if (1.0e-15 < dv)
			{
				fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", 0.0, dv);
			}
			else
			{
				fprintf(stderr, "V PASSED\n");
			}
		}

		delete sim_data;
	}

	// Test calc_radius()
	{
		char test_func[] = "calc_radius";
	
		var_t mass = 1.0;
		var_t density = 1.0;

		var_t expected = 0.62035049089940001666800681204778;
		var_t result = tools::calc_radius(mass, density);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected - result))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test calc_mass()
	{
		char test_func[] = "calc_mass";
	
		var_t radius = 1.0;
		var_t density = 1.0;

		var_t expected = 4.1887902047863909846168578443727;
		var_t result = tools::caclulate_mass(radius, density);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected - result))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test calc_density()
	{
		char test_func[] = "calc_density";
	
		var_t mass = 1.0;
		var_t radius = 1.0;

		var_t expected = 0.23873241463784300365332564505877;
		var_t result = tools::calc_density(mass, radius);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected - result))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test calc_position_after_collision()
	{
		char test_func[] = "calc_position_after_collision";
	
		var_t m1 = 1.0;
		var_t m2 = 1.0;

		vec_t r1 = {1.0, 1.0, 1.0, 0.0};
		vec_t r2 = {4.0, 3.0, 1.0, 0.0};

		vec_t expected = {2.5, 2.0, 1.0, 0.0};
		vec_t result = {0.0, 0.0, 0.0, 0.0};
		tools::calc_position_after_collision(m1, m2, &r1, &r2, result);

		var_t dr = fabs(expected.x - result.x) + 
			       fabs(expected.y - result.y) + 
				   fabs(expected.z - result.z) +
				   fabs(expected.w - result.w);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < dr)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", 0.0, dr);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test calc_velocity_after_collision()
	{
		char test_func[] = "calc_velocity_after_collision";
	
		var_t m1 = 1.0;
		var_t m2 = 1.0;

		vec_t v1 = { 1.0, 0.0, 0.0, 0.0};
		vec_t v2 = {-1.0, 0.0, 0.0, 0.0};

		vec_t expected = {0.0, 0.0, 0.0, 0.0};
		vec_t result = {0.0, 0.0, 0.0, 0.0};
		tools::calc_velocity_after_collision(m1, m2, &v1, &v2, result);

		var_t dr = fabs(expected.x - result.x) + 
			       fabs(expected.y - result.y) + 
				   fabs(expected.z - result.z) +
				   fabs(expected.w - result.w);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < dr)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", 0.0, dr);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		v1.x = -1.0;
		v1.y =  0.0;
		v1.z =  0.0;

		v2.x = -1.0;
		v2.y =  0.0;
		v2.z =  0.0;

		expected.x = -1.0;
		expected.y =  0.0;
		expected.z =  0.0;
		tools::calc_velocity_after_collision(m1, m2, &v1, &v2, result);

		dr = fabs(expected.x - result.x) + 
		     fabs(expected.y - result.y) + 
		     fabs(expected.z - result.z) +
			 fabs(expected.w - result.w);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < dr)
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", 0.0, dr);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test calc_physical_properties()
	{
		char test_func[] = "calc_physical_properties";

		param_t p1 = {0.0, 0.0, 0.0, 0.0};
		param_t p2 = {0.0, 0.0, 0.0, 0.0};

		p1.cd      = 1.0;
		p1.density = 1.0;
		p1.mass    = 1.0;
		p1.radius  = 1.0;

		p2.cd      = 1.0;
		p2.density = 1.0;
		p2.mass    = 1.0;
		p2.radius  = 1.0;

		param_t result = {0.0, 0.0, 0.0, 0.0};
		param_t expected = {0.0, 0.0, 0.0, 0.0};
		expected.mass = 2.0;
		expected.density = 0.23873241463784300365332564505877;
		expected.radius = tools::calc_radius(2.0, 0.23873241463784300365332564505877);
		expected.cd = 1.0;
		tools::calc_physical_properties(p1, p2, result);
		
		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected.cd - result.cd))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected.cd, result.cd);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected.density - result.density))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected.density, result.density);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected.mass - result.mass))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected.mass, result.mass);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-15 < fabs(expected.radius - result.radius))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected.radius, result.radius);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test kepler_equation_solver()
	{
		char test_func[] = "kepler_equation_solver";

		var_t ecc = 0.5;

		var_t mean_anomaly = 27.0 * constants::DegreeToRadian;
		var_t expected = 48.43417991487915; // degree
		var_t result = 0.0;
		tools::kepler_equation_solver(ecc, mean_anomaly, 1.0e-15, &result);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-13 < fabs(expected - (result * constants::RadianToDegree)))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result * constants::RadianToDegree);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		mean_anomaly = 225.0 * constants::DegreeToRadian;
		expected = 210.47211284530107; // degree
		result = 0.0;
		tools::kepler_equation_solver(ecc, mean_anomaly, 1.0e-15, &result);

		fprintf(stderr, "\t%s(): ", test_func);
		if (1.0e-13 < fabs(expected - (result * constants::RadianToDegree)))
		{
			fprintf(stderr, "FAILED\n\t\tExpected: %25.16lf\n\t\t But was: %25.16lf\n", expected, result * constants::RadianToDegree);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test print_vector()
	{
		char test_func[] = "print_vector";

		vec_t v = {-1.0, 1.0, 2.0, 0.0};

		fprintf(stderr, "\t%s(): [the two lines below must be identical]\n", test_func);
		fprintf(stderr, " -1.0000000000000000e+000  1.0000000000000000e+000  2.0000000000000000e+000  0.0000000000000000e+000\n");
		tools::print_vector(&v);
		fprintf(stderr, "\tPASSED ??\n");
	}
}

namespace red_test
{
	int run(int argc, char *argv[])
	{
		test_tools();
		test_util();

		return 0;
	}
} /* red_test */

} /* redutilcu */

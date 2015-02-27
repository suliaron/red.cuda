// includes system
#include <algorithm>
#include <iomanip>
#include <fstream>

// includes project
#include "file_util.h"
#include "tools.h"
#include "red_macro.h"
#include "red_constants.h"

#include "util.h"

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

}

static void test_tools()
{
	char test_set[] = "test_tools";

	//bool is_number(const string& str);
	//void trim_right(string& str);
	//void trim_right(string& str, char c);
	//void trim_left(string& str);
	//void trim(string& str);
	//string get_time_stamp();
	//string convert_time_t(time_t t);

	////! Computes the total mass of the system
	//var_t get_total_mass(int n, const sim_data_t *sim_data);
	////! Computes the total mass of the bodies with type in the system
	//var_t get_total_mass(int n, body_type_t type, const sim_data_t *sim_data);
	//void compute_bc(int n, bool verbose, const sim_data_t *sim_data, vec_t* R0, vec_t* V0);
	//void transform_to_bc(int n, bool verbose, const sim_data_t *sim_data);

	//var_t calculate_radius(var_t m, var_t density);
	//var_t calculate_density(var_t m, var_t R);
	//var_t caclulate_mass(var_t R, var_t density);

	//int	kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E);
	//int calculate_phase(var_t mu, const orbelem_t* oe, vec_t* rVec, vec_t* vVec);

	//void print_vector(vec_t *v);

	fprintf(stderr, "TEST: %s\n", test_set);

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

	// Test compute_bc()
	{
		char test_func[] = "compute_bc";
	
		sim_data_t* sim_data = create_sim_data();

		vec_t expected_R0 = {0.5, 0.5, 0.0, 0.0};
		vec_t expected_V0 = {1.5, 0.0, 0.0, 0.0};

		vec_t result_R0 = {0.0, 0.0, 0.0, 0.0};
		vec_t result_V0 = {0.0, 0.0, 0.0, 0.0};

		tools::compute_bc(4, false, sim_data, &result_R0, &result_V0);

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

	// Test transform_to_bc 
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

}


namespace red_test
{
	int run(int argc, char *argv[])
	{
		test_tools();

		return 0;
	}
} /* red_test */

} /* redutilcu */

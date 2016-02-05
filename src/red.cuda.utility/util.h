#pragma once

#include "red_type.h"

namespace redutilcu
{
	__host__ __device__ var4_t rotate_2D_vector(var_t theta, const var4_t& r);

	template <typename T>
	std::string number_to_string( T number, uint32_t width, bool fill);
	template <typename T>
	std::string number_to_string(T number);

	void device_query(ostream& sout, int id_dev, bool print_to_screen);

	int get_id_fastest_cuda_device();
	int get_n_cuda_device();
	std::string get_device_name(int id_dev);

	void set_kernel_launch_param(uint32_t n_data, uint16_t n_tpb, dim3& grid, dim3& block);

	void allocate_host_vector(  void **ptr, size_t size,           const char *file, int line);
	void allocate_device_vector(void **ptr, size_t size,           const char *file, int line);
	void allocate_vector(       void **ptr, size_t size, bool cpu, const char *file, int line);

	#define ALLOCATE_HOST_VECTOR(  ptr, size)      (allocate_host_vector(  ptr, size,      __FILE__, __LINE__))
	#define ALLOCATE_DEVICE_VECTOR(ptr, size)      (allocate_device_vector(ptr, size,      __FILE__, __LINE__))
	#define ALLOCATE_VECTOR(       ptr, size, cpu) (allocate_vector(       ptr, size, cpu, __FILE__, __LINE__))

	void free_host_vector(  void **ptr,           const char *file, int line);
	void free_device_vector(void **ptr,           const char *file, int line);
	void free_vector(       void **ptr, bool cpu, const char *file, int line);

	#define FREE_HOST_VECTOR(  ptr)      (free_host_vector(  ptr,      __FILE__, __LINE__))
	#define FREE_DEVICE_VECTOR(ptr)      (free_device_vector(ptr,      __FILE__, __LINE__))
	#define FREE_VECTOR(       ptr, cpu) (free_vector(       ptr, cpu, __FILE__, __LINE__))

	void allocate_host_storage(pp_disk_t::sim_data_t *sd, int n);
	void allocate_device_storage(pp_disk_t::sim_data_t *sd, int n);

	void deallocate_host_storage(pp_disk_t::sim_data_t *sd);
	void deallocate_device_storage(pp_disk_t::sim_data_t *sd);

	void copy_vector_to_device(void* dst, const void *src, size_t count);
	void copy_vector_to_host(  void* dst, const void *src, size_t count);
	void copy_vector_d2d(      void* dst, const void *src, size_t count);

	void copy_constant_to_device(const void* dst, const void *src, size_t count);


	void set_device(int id_of_target_dev, ostream& sout);
	void print_array(std::string path, int n, var_t *data, computing_device_t comp_dev);

	void create_aliases(computing_device_t comp_dev, pp_disk_t::sim_data_t *sd);
} /* redutilcu */

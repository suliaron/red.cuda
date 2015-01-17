#pragma once

#include "red_type.h"

int device_query(ostream& sout, int id_dev);

void allocate_vector(       void **ptr,   size_t size, bool cpu, const char *file, int line);
void allocate_host_vector(  void **h_ptr, size_t size,           const char *file, int line);
void allocate_device_vector(void **d_ptr, size_t size,           const char *file, int line);

#define ALLOCATE_DEVICE_VECTOR(d_ptr, size)      (allocate_device_vector(d_ptr, size,      __FILE__, __LINE__))
#define ALLOCATE_VECTOR(       ptr,   size, cpu) (allocate_vector(       ptr,   size, cpu, __FILE__, __LINE__))

void free_vector(       void *ptr, bool cpu, const char *file, int line);
void free_host_vector(  void *ptr,           const char *file, int line);
void free_device_vector(void *ptr,           const char *file, int line);

#define FREE_VECTOR(ptr, cpu) (free_vector(ptr, cpu, __FILE__, __LINE__))

void copy_vector_to_device(void* dst, const void *src, size_t count);
void copy_vector_to_host(void* dst, const void *src, size_t count);
void copy_vector_d2d(void* dst, const void *src, size_t count);

void copy_constant_to_device(const void* dst, const void *src, size_t count);

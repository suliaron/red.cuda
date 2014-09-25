#pragma once

#include "red_type.h"

void allocate_device_vector(void **d_ptr, size_t size, const char *file, int line);
#define ALLOCATE_DEVICE_VECTOR(d_ptr, size) (allocate_device_vector(d_ptr, size, __FILE__, __LINE__))

void copy_vector_to_device(void* dst, const void *src, size_t count);
void copy_vector_to_host(void* dst, const void *src, size_t count);

void copy_constant_to_device(const void* dst, const void *src, size_t count);

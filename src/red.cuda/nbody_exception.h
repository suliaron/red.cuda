#pragma once

#include "cuda_runtime.h"

#include <exception>
#include <string>

class nbody_exception : public std::exception
{
private:
	std::string message;
	cudaError_t cuda_error;
public:
	nbody_exception(std::string message);
	nbody_exception(std::string message, cudaError_t cuda_error);
	~nbody_exception() throw();

	const char* what() const throw();
};

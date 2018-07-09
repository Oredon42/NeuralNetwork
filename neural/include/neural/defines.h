#ifndef DEFINES_H
#define DEFINES_H

#include <vector>

#ifdef _SIMPLE_PRECISION
	using real = float;
#else
	using real = double;
#endif

using Input = real;
using Output = real;
using Weight = real;
using Error = real;

using Inputs = std::vector<Input>;
using Outputs = std::vector<Output>;
using Weights = std::vector<Weight>;
using Errors = std::vector<Error>;

#endif // DEFINES_H
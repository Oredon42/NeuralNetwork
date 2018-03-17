#ifndef DEFINES_H
#define DEFINES_H

#include <vector>

using real = double;

using Input = real;
using Output = real;
using Weight = real;

using Inputs = std::vector<Input>;
using Outputs = std::vector<Output>;
using Weights = std::vector<Weight>;

struct Dataset
{
    std::vector<Inputs> aInputs;
    std::vector<Outputs> aOutputs;
};

#endif // DEFINES_H
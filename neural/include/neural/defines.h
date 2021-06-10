#ifndef DEFINES_H
#define DEFINES_H

#ifdef _MSC_VER
    #define INLINE __forceinline
#elif __GCC__ or __clang__
    #define INLINE inline
#else
    #define INLINE
#endif

#include <vector>

#ifdef _SIMPLE_PRECISION
    using real = float;
#else
    using real = double;
#endif

using PerceptronInput = real;
using PerceptronOutput = real;
using PerceptronWeight = real;
using PerceptronError = real;

using PerceptronErrors = std::vector<real>;

using LayerInputs = std::vector<PerceptronInput>;
using LayerOutputs = std::vector<PerceptronOutput>;
using LayerWeights = std::vector<PerceptronWeight>;
using LayerErrors = std::vector<PerceptronErrors>;

extern unsigned int e_uiSeed;

#endif // DEFINES_H

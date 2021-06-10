#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "neural/defines.h"

#include <math.h>

enum class ActivationFunctionType
{
    Linear,
    HyperbolicTangent,
    RectifiedLinearUnits
};

// Rename for simplicity
using ActivationFunctionPtr = real (*)(real);
using ActivationDerivativePtr = real(*)(real);

// Getters
INLINE ActivationFunctionPtr activationFunctionFromType(ActivationFunctionType type);
INLINE ActivationDerivativePtr activationDerivativeFromType(ActivationFunctionType type);

// Functions declarations
INLINE real linear(real rValue);
INLINE real hyperbolicTangent(real rValue);
INLINE real rectifiedLinearUnits(real rValue);

// Derivatives declarations
INLINE real dLinear(real rValue);
INLINE real dHyperbolicTangent(real rValue);
INLINE real dRectifiedLinearUnits(real rValue);

// Getters

ActivationFunctionPtr activationFunctionFromType(ActivationFunctionType type)
{
    switch(type)
    {
    case ActivationFunctionType::Linear:
        return &linear;
    case ActivationFunctionType::HyperbolicTangent:
        return &hyperbolicTangent;
    case ActivationFunctionType::RectifiedLinearUnits:
        return &rectifiedLinearUnits;
    default:
        return nullptr;
    }
}

ActivationDerivativePtr activationDerivativeFromType(ActivationFunctionType type)
{
    switch (type)
    {
    case ActivationFunctionType::Linear:
        return &dLinear;
    case ActivationFunctionType::HyperbolicTangent:
        return &dHyperbolicTangent;
    case ActivationFunctionType::RectifiedLinearUnits:
        return &dRectifiedLinearUnits;
    default:
        return nullptr;
    }
}

// Functions definitions
real linear(real rValue)
{
    return rValue;
}

real hyperbolicTangent(real rValue)
{
    return tanh(rValue);
}

real rectifiedLinearUnits(real rValue)
{
    return fmax(rValue, 0.0);
}

// Derivatives definitions
real dLinear(real rValue)
{
    return 1.0;
}

real dHyperbolicTangent(real rValue)
{
    real tmp = tanh(rValue);
    return 1.0 - (tmp * tmp);
}

real dRectifiedLinearUnits(real rValue)
{
    return rValue > 0.0 ? 1.0 : 0.0;
}

#endif // ACTIVATION_FUNCTIONS_H

#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "neural/defines.h"

#include <math.h>

enum ActivationFunctionType
{
    HyperbolicTangent,
    RectifiedLinearUnits
};

// Rename for simplicity
using ActivationFunction = real (*)(const real &);
using ActivationDerivative = real(*)(const real &);
using ActivationBounds = bool(*)(real &, real &, real &, real &);

// Getters
INLINE ActivationFunction activationFunctionFromType(const ActivationFunctionType &type);
INLINE ActivationDerivative activationDerivativeFromType(const ActivationFunctionType &type);
INLINE ActivationBounds activationBoundsFromType(const ActivationFunctionType &type);

// Functions declarations
INLINE real hyperbolicTangent(const real &rValue);
INLINE real rectifiedLinearUnits(const real &rValue);

// Derivatives declarations
INLINE real dHyperbolicTangent(const real &rValue);
INLINE real dRectifiedLinearUnits(const real &rValue);

// Inputs/Outputs bounds declarations
INLINE bool hyperbolicTangentBounds(real &rInputsMin, real &rInputsMax, real &rOutputsMin, real &rOutputsMax);
INLINE bool rectifiedLinearUnitsBounds(real &rInputsMin, real &rInputsMax, real &rOutputsMin, real &rOutputsMax);

// Getters

ActivationFunction activationFunctionFromType(const ActivationFunctionType &type)
{
    switch(type)
    {
    case HyperbolicTangent:
        return &hyperbolicTangent;
    case RectifiedLinearUnits:
        return &rectifiedLinearUnits;
    default:
        return nullptr;
    }
}

ActivationDerivative activationDerivativeFromType(const ActivationFunctionType &type)
{
    switch (type)
    {
    case HyperbolicTangent:
        return &dHyperbolicTangent;
    case RectifiedLinearUnits:
        return &dRectifiedLinearUnits;
    default:
        return nullptr;
    }
}

ActivationBounds activationBoundsFromType(const ActivationFunctionType &type)
{
    switch (type)
    {
    case HyperbolicTangent:
        return &hyperbolicTangentBounds;
    case RectifiedLinearUnits:
        return &rectifiedLinearUnitsBounds;
    default:
        return nullptr;
    }
}


// Functions definitions
real hyperbolicTangent(const real &rValue)
{
    return tanh(rValue);
}

real rectifiedLinearUnits(const real &rValue)
{
    return fmax(rValue, 0.0);
}

// Derivatives definitions
real dHyperbolicTangent(const real &rValue)
{
    const real &tmp = tanh(rValue);
    return 1.0 - (tmp * tmp);
}

real dRectifiedLinearUnits(const real &rValue)
{
    return rValue > 0.0 ? 1.0 : 0.0;
}

// Inputs/Outputs bounds declarations
bool hyperbolicTangentBounds(real &rInputsMin, real &rInputsMax, real &rOutputsMin, real &rOutputsMax)
{
    rInputsMin = -5.0;
    rInputsMax = 5.0;
    rOutputsMin = -1.0;
    rOutputsMax = 1.0;
    return true;
}

bool rectifiedLinearUnitsBounds(real &rInputsMin, real &rInputsMax, real &rOutputsMin, real &rOutputsMax)
{
    return false; // No bounds for ReLu
}

#endif // ACTIVATION_FUNCTIONS_H

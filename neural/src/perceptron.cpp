#include "neural/perceptron.h"

#include "neural/assert.h"

#include <cmath>
#include <random>
#include <chrono>

Perceptron::Perceptron(const size_t &uiInputsSize, const PerceptronParameters &parameters) :
    m_aWeights(uiInputsSize),
    m_rLearningRate(parameters.rLearningRate),
    m_rBias(parameters.rBias),
    m_rMomentum(parameters.rMomentum),
    m_arSavedDerivatives(uiInputsSize)
{
    m_pfActivationFunctionPtr = activationFunctionFromType(parameters.eActivationFunctionType);
    m_pfActivationDerivativePtr = activationDerivativeFromType(parameters.eActivationFunctionType);
}

void Perceptron::evaluate(const LayerInputs &aInputs, PerceptronOutput &output) const
{
    output = m_pfActivationFunctionPtr(evaluationFunction(aInputs));
}

/*
 * Train Perceptron given training data
 * (Inputs + target Output)
 * Error is computed and returned for each Input,
 * which can be used for backpropagation.
 */
void Perceptron::train(const LayerInputs &aInputs, PerceptronOutput rTargetOutput, PerceptronErrors &aErrors)
{
    ASSERT(aInputs.size() == numberOfInputs());

    // Evaluate
    const real rZ = evaluationFunction(aInputs);
    const PerceptronOutput rActualOutput = m_pfActivationFunctionPtr(rZ);

    // Compute Error
    const PerceptronError rError = m_pfActivationDerivativePtr(rZ) * (rActualOutput - rTargetOutput);

    for(size_t i = 0; i < numberOfInputs(); ++i)
    {
        const real rDerivative = rError * aInputs[i];

        // Compute error to send to previous layers (backpropagation)
        aErrors[i] = rError * m_aWeights[i];

        // Update weights
        m_aWeights[i] -= rDerivative * m_rLearningRate + m_arSavedDerivatives[i] * m_rMomentum;

        // Save derivative
        m_arSavedDerivatives[i] = rDerivative;
    }
}

/*
 * Train Perceptron inside a hidden layer or
 * inputs layer of a Neural Network,
 * given Inputs and Errors of Perceptron of
 * the next Layer and the index of Perceptron
 * in current Layer.
 */
void Perceptron::train(const LayerInputs &aInputs, const LayerErrors &aNextLayerErrors, size_t nodeIndex, PerceptronErrors &aErrors)
{
    ASSERT(aInputs.size() == numberOfInputs());

    // Evaluate
    const real &rZ = evaluationFunction(aInputs);

    // Compute sum of next Layer errors
    real rWeightedNextLayerErrorSum = 0.0;
    for(size_t i = 0; i < aNextLayerErrors.size(); ++i)
    {
        rWeightedNextLayerErrorSum += aNextLayerErrors[i][nodeIndex];
    }
    
    // Compute Error
    const PerceptronError rError = m_pfActivationDerivativePtr(rZ) * rWeightedNextLayerErrorSum;

    for(size_t i = 0; i < numberOfInputs(); ++i)
    {
        const real rDerivative = rError * aInputs[i];

        // Compute error to send to previous layers (backpropagation)
        aErrors[i] = rError * m_aWeights[i];

        // Update weights
        m_aWeights[i] -= rDerivative * m_rLearningRate + m_arSavedDerivatives[i] * m_rMomentum;

        // Save derivative
        m_arSavedDerivatives[i] = rDerivative;
    }
}

void Perceptron::initializeRandomWeights()
{
    std::uniform_real_distribution<real> unif(0.0, 1.0);
    std::default_random_engine re(e_uiSeed++);

    const real epsilon = 2.4494897427831780981972840747059 / sqrt(static_cast<real>(numberOfInputs() + 1.0));

    for(size_t i = 0; i < numberOfInputs(); ++i)
    {
        m_aWeights[i] = unif(re) * (2 * epsilon) - epsilon;
    }
}

void Perceptron::setActivationFunction(ActivationFunctionType eActivationFunctionType)
{
    m_pfActivationFunctionPtr = activationFunctionFromType(eActivationFunctionType);
    m_pfActivationDerivativePtr = activationDerivativeFromType(eActivationFunctionType);
}

void Perceptron::setInputsSize(size_t uiInputsSize)
{
    m_aWeights.resize(uiInputsSize);
    m_arSavedDerivatives.resize(uiInputsSize);
}
void Perceptron::setLearningRate(real rLearningRate)
{
    m_rLearningRate = rLearningRate;
}

void Perceptron::setBias(real rBias)
{
    m_rBias = rBias;
}

/*
 * Dot product of inputs and weights
 */
real Perceptron::evaluationFunction(const LayerInputs &aInputs) const
{
    ASSERT(aInputs.size() == numberOfInputs());

    real rZ = m_rBias;

    for(size_t i = 0; i < numberOfInputs(); ++i)
    {
        rZ += m_aWeights[i] * aInputs[i];
    }

    return rZ;
}

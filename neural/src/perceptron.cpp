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
    m_pfActivationFunction = activationFunctionFromType(parameters.eActivationFunctionType);
    m_pfActivationDerivative = activationDerivativeFromType(parameters.eActivationFunctionType);
}

Output Perceptron::evaluate(const Inputs &aInputs) const
{
    return m_pfActivationFunction(evaluationFunction(aInputs));
}

/*
 * Train Perceptron given training data
 * (Inputs + target Output)
 * Error is computed and returned for each Input,
 * which can be used for backpropagation.
 */
Errors Perceptron::train(const Inputs &aInputs, const Output &rTargetOutput)
{
    ASSERT(aInputs.size() == numberOfInputs());

    // Evaluate
    const real &rZ = evaluationFunction(aInputs);
    const Output &rActualOutput = m_pfActivationFunction(rZ);

    // Compute Error
    const Error rError = m_pfActivationDerivative(rZ) * (rActualOutput - rTargetOutput);
    Errors aOutputErrors(numberOfInputs());

    for(size_t i = 0; i < numberOfInputs(); ++i)
    {
        const real rDerivative = rError * aInputs[i];

        // Compute error to send to previous layers (backpropagation)
        aOutputErrors[i] = rError * m_aWeights[i];

        // Update weights
        m_aWeights[i] -= rDerivative * m_rLearningRate + m_arSavedDerivatives[i] * m_rMomentum;

        // Save derivative
        m_arSavedDerivatives[i] = rDerivative;
    }

    return aOutputErrors;
}

/*
 * Train Perceptron inside a hidden layer or
 * inputs layer of a Neural Network,
 * given Inputs and Errors of Perceptron of
 * the next Layer and the index of Perceptron
 * in current Layer.
 */
Errors Perceptron::train(const Inputs &aInputs, const std::vector<Errors> &aNextLayerErrors, const size_t &nodeIndex)
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
    const Error rError = m_pfActivationDerivative(rZ) * rWeightedNextLayerErrorSum;
    Errors aOutputErrors(numberOfInputs());

    for(size_t i = 0; i < numberOfInputs(); ++i)
    {
        const real rDerivative = rError * aInputs[i];

        // Compute error to send to previous layers (backpropagation)
        aOutputErrors[i] = rError * m_aWeights[i];

        // Update weights
        m_aWeights[i] -= rDerivative * m_rLearningRate + m_arSavedDerivatives[i] * m_rMomentum;

        // Save derivative
        m_arSavedDerivatives[i] = rDerivative;
    }

    return aOutputErrors;
}

void Perceptron::initializeRandomWeights()
{
    std::uniform_real_distribution<real> unif(0.0, 1.0);
    const unsigned int &uiSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine re(uiSeed);

    const real epsilon = 2.4494897427831780981972840747059 / sqrt(static_cast<real>(numberOfInputs() + 1.0));

    for(size_t i = 0; i < numberOfInputs(); ++i)
    {
        m_aWeights[i] = unif(re) * (2 * epsilon) - epsilon;
    }
}

void Perceptron::setActivationFunction(const ActivationFunctionType &eActivationFunctionType)
{
    m_pfActivationFunction = activationFunctionFromType(eActivationFunctionType);
    m_pfActivationDerivative = activationDerivativeFromType(eActivationFunctionType);
}

void Perceptron::setInputsSize(const size_t &uiInputsSize)
{
    m_aWeights.resize(uiInputsSize);
    m_arSavedDerivatives.resize(uiInputsSize);
}
void Perceptron::setLearningRate(const real &rLearningRate)
{
    m_rLearningRate = rLearningRate;
}

void Perceptron::setBias(const real &rBias)
{
    m_rBias = rBias;
}

/*
 * Dot product of inputs and weights
 */
real Perceptron::evaluationFunction(const Inputs &aInputs) const
{
    ASSERT(aInputs.size() == numberOfInputs());

    real rZ = m_rBias;

    for(size_t i = 0; i < numberOfInputs(); ++i)
    {
        rZ += m_aWeights[i] * aInputs[i];
    }

    return rZ;
}

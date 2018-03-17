#include "perceptron.h"

#include <cmath>
#include <random>
#include <chrono>

Perceptron::Perceptron(const std::size_t &inputsSize)
{
    if(inputsSize > 0)
    {
        m_aWeights.resize(inputsSize);
    }
}

Perceptron::~Perceptron()
{
}

Output Perceptron::evaluate(const Inputs &aInputs) const
{
    if(aInputs.size() == m_aWeights.size())
    {
        real rZ = m_rBias;

        for(std::size_t i = 0; i < m_aWeights.size(); ++i)
        {
            rZ += m_aWeights[i] * aInputs[i];
        }

        if(m_eActivationFunction == Heaviside)
        {
            return((rZ > 0.0)?1.0:0.0);
        }
        else
        {
            return tanh(rZ);
        }
    }
}

void Perceptron::train(const Inputs &aInputs, const Output &wantedOuput)
{
    if(aInputs.size() == m_aWeights.size())
    {
        const Output &output = evaluate(aInputs);
        const real rError = m_rLearningRate * (wantedOuput - output);

        for(std::size_t i = 0; i < m_aWeights.size(); ++i)
        {
            m_aWeights[i] += aInputs[i] * rError;
        }
    }
}

void Perceptron::randomizeWeights()
{
    std::uniform_real_distribution<real> unif(0.0, 1.0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine re(seed);

    for(std::size_t i = 0; i < m_aWeights.size(); ++i)
    {
        m_aWeights[i] = unif(re) - 0.5;
    }
}

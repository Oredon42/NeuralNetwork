#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "neural/defines.h"
#include "neural/activation_functions.h"

class Perceptron
{
public:
    struct PerceptronParameters
    {
        ActivationFunctionType eActivationFunctionType;
        real rLearningRate;
        real rBias;
        real rMomentum;
    };

public:
    Perceptron(const size_t &uiInputsSize, const PerceptronParameters &parameters);

    Output evaluate(const Inputs &aInputs) const;

    Errors train(const Inputs &aInputs, const Output &rTargetOutput);
    Errors train(const Inputs &aInputs, const std::vector<Errors> &aNextLayerErrors, const size_t &nodeIndex);

    void initializeRandomWeights();

    // Setters
    void setActivationFunction(const ActivationFunctionType &eActivationFunctionType);
    void setInputsSize(const size_t &uiInputsSize);
    void setLearningRate(const real &rLearningRate);
    void setBias(const real &rBias);

    // Getters
    INLINE size_t numberOfInputs() const{return m_aWeights.size();}
    INLINE real learningRate() const{return m_rLearningRate;}
    INLINE real bias() const{return m_rBias;}

private:
    ActivationFunction m_pfActivationFunction = nullptr;
    ActivationDerivative m_pfActivationDerivative = nullptr;

    Weights m_aWeights;

    real m_rLearningRate = 0.1;
    real m_rBias = 0.0;

    real m_rMomentum = 0.3;
    std::vector<real> m_arSavedDerivatives;

private:
    // Private methods
    real evaluationFunction(const Inputs &aInputs) const;
};

#endif // PERCEPTRON_H

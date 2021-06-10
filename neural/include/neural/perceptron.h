#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "neural/defines.h"
#include "neural/activation_functions.h"

struct PerceptronParameters
{
    ActivationFunctionType eActivationFunctionType;
    real rLearningRate;
    real rBias;
    real rMomentum;
};

class Perceptron
{
public:
    Perceptron(const size_t &uiInputsSize, const PerceptronParameters &parameters);

    void evaluate(const LayerInputs &aInputs, PerceptronOutput &output) const;

    void train(const LayerInputs &aInputs, PerceptronOutput rTargetOutput, PerceptronErrors &aErrors);
    void train(const LayerInputs &aInputs, const LayerErrors &aNextLayerErrors, size_t nodeIndex, PerceptronErrors &aErrors);

    void initializeRandomWeights();

    // Setters
    void setActivationFunction(ActivationFunctionType eActivationFunctionType);
    void setInputsSize(size_t uiInputsSize);
    void setLearningRate(real rLearningRate);
    void setBias(real rBias);

    // Getters
    INLINE size_t numberOfInputs() const{return m_aWeights.size();}
    INLINE real learningRate() const{return m_rLearningRate;}
    INLINE real bias() const{return m_rBias;}

private:
    ActivationFunctionPtr m_pfActivationFunctionPtr = nullptr;
    ActivationDerivativePtr m_pfActivationDerivativePtr = nullptr;

    LayerWeights m_aWeights;

    real m_rLearningRate = 0.1;
    real m_rBias = 0.0;

    real m_rMomentum = 0.3;
    std::vector<real> m_arSavedDerivatives;

private:
    // Private methods
    real evaluationFunction(const LayerInputs &aInputs) const;
};

#endif // PERCEPTRON_H

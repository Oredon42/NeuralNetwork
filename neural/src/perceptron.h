#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "defines.h"

class Perceptron
{
public:
    enum ActivationFunction
    {
        Heaviside,
        HyperbolicTangent
    };

    Perceptron(const std::size_t &inputsSize = 1);
    ~Perceptron();

    Output evaluate(const Inputs &aInputs) const;
    void train(const Inputs &aInputs, const Output &wantedOuput);
    void randomizeWeights();

    // Setters
    inline void setActivationFunction(const ActivationFunction &eActivationFunction) {m_eActivationFunction = eActivationFunction;}
    inline void setInputsSize(const std::size_t &inputsSize) {if(inputsSize > 0) m_aWeights.resize(inputsSize);}
    inline void setLearningRate(const real &rLearningRate) {m_rLearningRate = rLearningRate;}
    inline void setBias(const real &rBias) {m_rBias = rBias;}

    // Getters
    inline ActivationFunction activationFunction() const{return m_eActivationFunction;}
    inline std::size_t inputsSize() const{return m_aWeights.size();}
    inline real learningRate() const{return m_rLearningRate;}
    inline real bias() const{return m_rBias;}

private:
    ActivationFunction m_eActivationFunction = Heaviside;

    Weights m_aWeights;

    real m_rLearningRate = 0.1;
    real m_rBias = 0.0;
};

#endif // PERCEPTRON_H
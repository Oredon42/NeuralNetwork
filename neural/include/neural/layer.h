#ifndef LAYER_H
#define LAYER_H

#include "neural/perceptron.h"

struct LayerParameters
{
    size_t layerSize;
    PerceptronParameters perceptronParameters;
};

class Layer
{
public:
    Layer(const size_t &previousLayerSize, const LayerParameters &parameters);

    void evaluate(const LayerInputs &aInputs, LayerOutputs &aOutputs) const;

    void train(const LayerInputs &aInputs, const LayerOutputs &aTargetOutputs, LayerErrors &aErrors);
    void train(const LayerInputs &aInputs, const LayerErrors &aNextLayerErrors, LayerErrors &aErrors);

    INLINE size_t size() const{return m_aPerceptrons.size();}
    INLINE ActivationFunctionType activationFunctionType() const{return m_eActivationFunctionType;}

private:
    ActivationFunctionType m_eActivationFunctionType;
    std::vector<Perceptron> m_aPerceptrons;
};

#endif // LAYER_H

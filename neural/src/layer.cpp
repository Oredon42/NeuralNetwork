#include "neural/layer.h"

#include "neural/assert.h"

Layer::Layer(const size_t &previousLayerSize, const LayerParameters &parameters)
{
    ASSERT(parameters.layerSize > 0);
    
    m_eActivationFunctionType = parameters.perceptronParameters.eActivationFunctionType;

    m_aPerceptrons.reserve(parameters.layerSize);

    for(size_t i = 0; i < parameters.layerSize; ++i)
    {
        m_aPerceptrons.push_back(Perceptron(previousLayerSize, parameters.perceptronParameters));
        m_aPerceptrons[i].initializeRandomWeights();
    }
}

void Layer::evaluate(const LayerInputs &aInputs, LayerOutputs &aOutputs) const
{
    ASSERT(aInputs.size() == m_aPerceptrons[0].numberOfInputs());

    for(size_t i = 0; i < size(); ++i)
    {
        m_aPerceptrons[i].evaluate(aInputs, aOutputs[i]);
    }
}

void Layer::train(const LayerInputs &aInputs, const LayerOutputs &aTargetOutputs, LayerErrors &aErrors)
{
    for(size_t i = 0; i < size(); ++i)
    {
        m_aPerceptrons[i].train(aInputs, aTargetOutputs[i], aErrors[i]);
    }
}

void Layer::train(const LayerInputs &aInputs, const LayerErrors &aNextLayerErrors, LayerErrors &aErrors)
{
    for(size_t i = 0; i < size(); ++i)
    {
        m_aPerceptrons[i].train(aInputs, aNextLayerErrors, i, aErrors[i]);
    }
}

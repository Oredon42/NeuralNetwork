#include "neural/multilayer_perceptron.h"

#include "neural/assert.h"

MultilayerPerceptron::MultilayerPerceptron(const MultilayerPerceptronParameters &parameters)
{
    ASSERT(parameters.aLayerParameters.size() > 0 && parameters.numberOfInputs > 0);

    m_aLayers.reserve(parameters.aLayerParameters.size());
    m_aOutputs.resize(parameters.aLayerParameters.size());
    m_aErrors.resize(parameters.aLayerParameters.size());

    m_numberOfInputs = parameters.numberOfInputs;
    size_t previousLayerSize = m_numberOfInputs;

    for(size_t i = 0; i < parameters.aLayerParameters.size(); ++i)
    {
        m_aLayers.push_back(Layer(previousLayerSize, parameters.aLayerParameters[i]));
        previousLayerSize = m_aLayers[i].size();
        m_aOutputs[i].resize(parameters.aLayerParameters[i].layerSize);
        m_aErrors[i].resize(parameters.aLayerParameters[i].layerSize);
        for(size_t j = 0; j < parameters.aLayerParameters[i].layerSize; ++j)
        {
            if(i > 0)
            {
                m_aErrors[i][j].resize(parameters.aLayerParameters[i - 1].layerSize);
            }
            else
            {
                m_aErrors[i][j].resize(parameters.numberOfInputs);
            }
        }
    }
}

const LayerOutputs &MultilayerPerceptron::evaluate(const LayerInputs &aInputs)
{
    ASSERT(aInputs.size() == m_numberOfInputs);

    // Inputs layer
    m_aLayers[0].evaluate(aInputs, m_aOutputs[0]);

    // Hidden layers + Output layer
    for(size_t i = 1; i < m_aLayers.size(); ++i)
    {
        m_aLayers[i].evaluate(m_aOutputs[i - 1], m_aOutputs[i]);
    }

    return m_aOutputs.back();
}

void MultilayerPerceptron::train(const LayerInputs &aInputs, const LayerOutputs &aTargetOuputs)
{
    ASSERT(aInputs.size() == m_numberOfInputs && aTargetOuputs.size() == m_aLayers.back().size());

    /*
     * Forward propagation
     */

    // Inputs layer
    m_aLayers[0].evaluate(aInputs, m_aOutputs[0]);

    // Hidden layers + Output layer
    for(size_t i = 1; i < m_aLayers.size(); ++i)
    {
        m_aLayers[i].evaluate(m_aOutputs[i - 1], m_aOutputs[i]);
    }

    /*
     * Back propagation
     */

    // Outputs layer
    size_t i = m_aLayers.size() - 1;
    m_aLayers.back().train(m_aOutputs[m_aOutputs.size() - 2], aTargetOuputs, m_aErrors[i]);

    // Hidden layers
    for(size_t i = m_aLayers.size() - 2; i > 0; --i)
    {
        m_aLayers[i].train(m_aOutputs[i - 1], m_aErrors[i + 1], m_aErrors[i]);
    }

    // Inputs layer
    m_aLayers.front().train(aInputs, m_aErrors[1], m_aErrors[0]);
}

Layer MultilayerPerceptron::layer(const size_t &i) const
{
    ASSERT(i <= m_aLayers.size());
    return m_aLayers[i];
}

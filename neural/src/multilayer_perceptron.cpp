#include "neural/multilayer_perceptron.h"

#include "neural/assert.h"

MultilayerPerceptron::MultilayerPerceptron(const MultilayerPerceptronParameters &parameters)
{
    ASSERT(parameters.aLayerParameters.size() > 0 && parameters.numberOfInputs > 0);

    m_aLayers.reserve(parameters.aLayerParameters.size());

    m_numberOfInputs = parameters.numberOfInputs;
    size_t previousLayerSize = m_numberOfInputs;

    for(size_t i = 0; i < parameters.aLayerParameters.size(); ++i)
    {
        m_aLayers.push_back(Layer(previousLayerSize, parameters.aLayerParameters[i]));
        previousLayerSize = m_aLayers[i].size();
    }
}

Outputs MultilayerPerceptron::evaluate(const Inputs &aInputs) const
{
    ASSERT(aInputs.size() == m_numberOfInputs);
    
    std::vector<Outputs> aEvaluationData(m_aLayers.size());

    // Inputs layer
    aEvaluationData[0] = m_aLayers[0].evaluate(aInputs);

    // Hidden layers + Output layer
    for(size_t i = 1; i < m_aLayers.size(); ++i)
    {
        aEvaluationData[i] = m_aLayers[i].evaluate(aEvaluationData[i - 1]);
    }

    return aEvaluationData.back();
}

void MultilayerPerceptron::train(const Inputs &aInputs, const Outputs &aTargetOuputs)
{
    ASSERT(aInputs.size() == m_numberOfInputs && aTargetOuputs.size() == m_aLayers.back().size());

    /*
     * Forward propagation
     */

    std::vector<Outputs> aEvaluationData(m_aLayers.size());

    // Inputs layer
    aEvaluationData[0] = m_aLayers[0].evaluate(aInputs);

    // Hidden layers + Output layer
    for(size_t i = 1; i < m_aLayers.size(); ++i)
    {
        aEvaluationData[i] = m_aLayers[i].evaluate(aEvaluationData[i - 1]);
    }

    /*
     * Back propagation
     */

    std::vector<std::vector<Errors>> aErrorsData(m_aLayers.size());

    // Outputs layer
    size_t i = m_aLayers.size() - 1;
    aErrorsData[i] = m_aLayers.back().train(aEvaluationData[aEvaluationData.size() - 2], aTargetOuputs);

    // Hidden layers
    for(size_t i = m_aLayers.size() - 2; i > 0; --i)
    {
        aErrorsData[i] = m_aLayers[i].train(aEvaluationData[i - 1], aErrorsData[i + 1]);
    }

    // Inputs layer
    m_aLayers.front().train(aInputs, aErrorsData[1]);
}

Layer MultilayerPerceptron::layer(const size_t &i) const
{
    ASSERT(i <= m_aLayers.size());
    return m_aLayers[i];
}

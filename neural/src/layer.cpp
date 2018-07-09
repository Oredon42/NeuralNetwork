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

Outputs Layer::evaluate(const Inputs &aInputs) const
{
	ASSERT(aInputs.size() == m_aPerceptrons[0].numberOfInputs());

	Outputs aOutputs(size());

	for(size_t i = 0; i < size(); ++i)
	{
		aOutputs[i] = m_aPerceptrons[i].evaluate(aInputs);
	}

	return aOutputs;
}

std::vector<Errors> Layer::train(const Inputs &aInputs, const Outputs &aTargetOutputs)
{
	std::vector<Errors> aErrors(size());

	for(size_t i = 0; i < size(); ++i)
	{
		aErrors[i] = m_aPerceptrons[i].train(aInputs, aTargetOutputs[i]);
	}

	return aErrors;
}

std::vector<Errors> Layer::train(const Inputs &aInputs, const std::vector<Errors> &aNextLayerErrors)
{
	std::vector<Errors> aErrors(size());

	for(size_t i = 0; i < size(); ++i)
	{
		aErrors[i] = m_aPerceptrons[i].train(aInputs, aNextLayerErrors, i);
	}

	return aErrors;
}

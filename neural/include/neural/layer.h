#ifndef LAYER_H
#define LAYER_H

#include "neural/perceptron.h"

class Layer
{
public:
	struct LayerParameters
	{
		size_t layerSize;
		Perceptron::PerceptronParameters perceptronParameters;
	};

public:
	Layer(const size_t &previousLayerSize, const LayerParameters &parameters);

	Outputs evaluate(const Inputs &aInputs) const;

	std::vector<Errors> train(const Inputs &aInputs, const Outputs &aTargetOutputs);
	std::vector<Errors> train(const Inputs &aInputs, const std::vector<Errors> &aNextLayerErrors);

	__forceinline size_t size() const{return m_aPerceptrons.size();}
	__forceinline ActivationFunctionType activationFunctionType() const{return m_eActivationFunctionType;}

private:
	ActivationFunctionType m_eActivationFunctionType;
	std::vector<Perceptron> m_aPerceptrons;
};

#endif // LAYER_H
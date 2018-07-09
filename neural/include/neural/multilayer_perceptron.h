#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include "neural/layer.h"

class MultilayerPerceptron
{
public:
	struct MultilayerPerceptronParameters
	{
		size_t numberOfInputs;
		std::vector<Layer::LayerParameters> aLayerParameters;
	};

public:
	MultilayerPerceptron(const MultilayerPerceptronParameters &parameters);

	Outputs evaluate(const Inputs &aInputs) const;
	void train(const Inputs &aInputs, const Outputs &aTargetOuputs);

	Layer layer(const size_t &i) const;

private:
	std::vector<Layer> m_aLayers;

	size_t m_numberOfInputs;
};

#endif // MULTILAYER_PERCEPTRON_H
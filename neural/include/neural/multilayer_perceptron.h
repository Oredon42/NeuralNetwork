#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include "neural/layer.h"

struct MultilayerPerceptronParameters
{
    size_t numberOfInputs;
    std::vector<LayerParameters> aLayerParameters;
};

class MultilayerPerceptron
{
public:
    MultilayerPerceptron(const MultilayerPerceptronParameters &parameters);

    const LayerOutputs &evaluate(const LayerInputs &aInputs);
    void train(const LayerInputs &aInputs, const LayerOutputs &aTargetOuputs);

    Layer layer(const size_t &i) const;

private:
    std::vector<Layer> m_aLayers;
    std::vector<LayerInputs> m_aOutputs;
    std::vector<LayerErrors> m_aErrors;
    size_t m_numberOfInputs;
};

#endif // MULTILAYER_PERCEPTRON_H

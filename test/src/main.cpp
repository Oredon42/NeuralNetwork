#include <iostream>

#include "neural/multilayer_perceptron.h"
#include "neural/trainer.h"
#include "neural/defines.h"
#include "datasetgenerator.h"

/*
 * Generation function
 * */
void regressionFunction(const LayerInputs &aInputs, LayerOutputs &aOutputs)
{
    aOutputs[0] = 2 * aInputs[0] + aInputs[1];
}

int main(int argc, char *argv[])
{
    size_t datasetSize = 0;

    if(argc != 2)
    {
        std::cout << "Usage: Neural <dataset_size>" << std::endl;
        return 1;
    }

    datasetSize = static_cast<size_t>(atoi(argv[1]));

    if(datasetSize <= 0)
    {
        return 1;
    }

    // Init Multilayer Perceptron
    PerceptronParameters perceptronParameters = { ActivationFunctionType::HyperbolicTangent, 0.1, 0.0, 0.01 };
    MultilayerPerceptronParameters multilayerPerceptronParameters;
    multilayerPerceptronParameters.numberOfInputs = 2;
    multilayerPerceptronParameters.aLayerParameters.push_back({ 5, perceptronParameters });
    multilayerPerceptronParameters.aLayerParameters.push_back({ 5, perceptronParameters });
    multilayerPerceptronParameters.aLayerParameters.push_back({ 1, perceptronParameters });

    MultilayerPerceptron multilayerPerceptron(multilayerPerceptronParameters);

    // Init Dataset
    Dataset dataset = DatasetGenerator::generateRandomDataset(datasetSize, -100.0, 100.0, multilayerPerceptronParameters, &regressionFunction);

    // Init Trainer
    TrainingParameters trainingParameters;
    trainingParameters.rErrorThreshold = 1e-6;
    trainingParameters.rCrossValidationEvaluationPercent = 0.7;
    trainingParameters.eScalingMethod = ScalingMethod::Normalisation;
    Trainer trainer(trainingParameters, true);

    // Start training
    trainer.train(multilayerPerceptron, dataset);

    return 0;
}

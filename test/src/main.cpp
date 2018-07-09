#include <iostream>

#include "neural/multilayer_perceptron.h"
#include "neural/trainer.h"
#include "datasetgenerator.h"

/*
 * Generation function
 * */
std::vector<double> regressionFunction(const std::vector<double> &inputs)
{
	std::vector<double> out(1);
	out[0] = inputs[0] * inputs[1];
	return out;
}

int main(int argc, char *argv[])
{
	size_t datasetSize = 0;

    if(argc != 3)
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
	Perceptron::PerceptronParameters perceptronParameters = { HyperbolicTangent, 0.1, 0.0 };
	MultilayerPerceptron::MultilayerPerceptronParameters multilayerPerceptronParameters;
	multilayerPerceptronParameters.numberOfInputs = 2;
	multilayerPerceptronParameters.aLayerParameters.push_back({ 10, perceptronParameters });
	multilayerPerceptronParameters.aLayerParameters.push_back({ 10, perceptronParameters });
	multilayerPerceptronParameters.aLayerParameters.push_back({ 1, perceptronParameters });
	MultilayerPerceptron multilayerPerceptron(multilayerPerceptronParameters);

	// Init Dataset
	Dataset dataset = DatasetGenerator::generateRandomDataset(datasetSize, -100.0, 100.0, &regressionFunction);

	// Init Trainer
	Trainer::TrainingParameters trainingParameters;
	trainingParameters.rMaxError = 0.0001;
	trainingParameters.rCrossValidationEvaluationPercent = 0.7;
	Trainer trainer(trainingParameters, true);

	// Start training
	trainer.train(multilayerPerceptron, dataset);

	getchar();

    return 0;
}
#include <iostream>

#include "src\perceptron.h"
#include "datasetgenerator.h"
#include "statistics.h"

/*
 * Generation function
 * */
Outputs classificationFunction(Inputs inputs)
{
	Outputs out;
	if(inputs.size() > 0)
	{
		out.push_back(static_cast<Output>(inputs[0] > 0));
	}
    return out;
}

Outputs regressionFunction(Inputs inputs)
{
	Outputs out;
	if(inputs.size() > 0)
	{
		out.push_back(static_cast<Output>(inputs[0] * inputs[0]));
	}
	return out;
}

/*
 * Perceptron evaluation routine
 * */
void perceptronEvaluationRoutine(Perceptron p, const Dataset &sEvaluationDataset,  const Dataset &sTrainingDataset)
{
    // Evaluation
    std::vector<Outputs> evaluatedOutputs(sEvaluationDataset.aInputs.size());

    for(std::size_t i = 0; i < evaluatedOutputs.size(); ++i)
    {
        evaluatedOutputs[i].push_back(p.evaluate(sEvaluationDataset.aInputs[i]));
    }
    std::cout << "Before training:" << std::endl;
    Statistics::printStatistics(evaluatedOutputs, sTrainingDataset.aOutputs);

    // Training
    for(std::size_t i = 0; i < sTrainingDataset.aInputs.size(); ++i)
    {
        p.train(sTrainingDataset.aInputs[i], sTrainingDataset.aOutputs[i][0]);
    }

    // Evaluation
    for(std::size_t i = 0; i < evaluatedOutputs.size(); ++i)
    {
		evaluatedOutputs[i].clear();
        evaluatedOutputs[i].push_back(p.evaluate(sEvaluationDataset.aInputs[i]));
    }
    std::cout << "After training:" << std::endl;
    Statistics::printStatistics(evaluatedOutputs, sEvaluationDataset.aOutputs);
}

int main(int argc, char *argv[])
{
    std::size_t evaluationDataSize = 0;
    std::size_t trainingDataSize = 0;

    if(argc != 3)
    {
        std::cout << "Usage: Neural <evaluation_size> <training_size>" << std::endl;
        return 1;
    }

    evaluationDataSize = static_cast<std::size_t>(atoi(argv[1]));
    trainingDataSize = static_cast<std::size_t>(atoi(argv[2]));
    
    if(evaluationDataSize == 0 || trainingDataSize == 0)
    {
        return 1;
    }

    Perceptron p(1);

    // Classification

	p.setActivationFunction(Perceptron::Heaviside);
    std::cout << "Classification:" << std::endl << std::endl;
    Dataset sClassificationEvaluationDataset = DatasetGenerator::generateRandomDataset(evaluationDataSize, -1.0, 1.0, &classificationFunction);
    Dataset sClassificationTrainingDataset = DatasetGenerator::generateRandomDataset(trainingDataSize, -1.0, 1.0, &classificationFunction);
    perceptronEvaluationRoutine(p, sClassificationEvaluationDataset, sClassificationTrainingDataset);

    // Regression

    p.setActivationFunction(Perceptron::HyperbolicTangent);
    std::cout << std::endl << "Regression:" << std::endl << std::endl;
    Dataset sRegressionEvaluationDataset = DatasetGenerator::generateRandomDataset(evaluationDataSize, 0.0, 1.0, &regressionFunction);
    Dataset sRegressionTrainingDataset = DatasetGenerator::generateRandomDataset(trainingDataSize, 0.0, 1.0, &regressionFunction);
    perceptronEvaluationRoutine(p, sRegressionEvaluationDataset, sRegressionTrainingDataset);

    return 0;
}
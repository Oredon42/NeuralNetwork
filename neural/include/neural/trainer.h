#ifndef TRAINER_H
#define TRAINER_H

#include "neural/defines.h"

class MultilayerPerceptron;
class Dataset;

class Trainer
{
private:
	static constexpr unsigned long s_maxIterations = 10000;

public:
	struct TrainingParameters
	{
		real rMaxError;
		real rCrossValidationEvaluationPercent;
	};

public:
	Trainer(const TrainingParameters &parameters, const bool &bVerbose = false);

	void train(MultilayerPerceptron &multilayerPerceptron, Dataset &dataset) const;

private:
	real m_rMaxError;
	real m_rCrossValidationEvaluationPercent;

	bool m_bVerbose;
};

#endif // TRAINER_H
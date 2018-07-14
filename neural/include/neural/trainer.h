#ifndef TRAINER_H
#define TRAINER_H

#include "neural/defines.h"

class MultilayerPerceptron;
class Dataset;

class Trainer
{
public:
    struct TrainingParameters
    {
        int iMaxIterations = -1;
        real rErrorThreshold = -1.0;
        real rTrainingRateThreshold = -1.0;
        real rCrossValidationEvaluationPercent = 1.0;
    };

public:
    Trainer(const TrainingParameters &parameters, const bool &bVerbose = false);

    void train(MultilayerPerceptron &multilayerPerceptron, Dataset &dataset) const;

private:
    int m_iMaxIterations;
    real m_rErrorThreshold;
    real m_rTrainingRateThreshold;
    real m_rCrossValidationEvaluationPercent;

    bool m_bVerbose;
};

#endif // TRAINER_H

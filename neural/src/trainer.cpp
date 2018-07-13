#include "neural/trainer.h"

#include "neural/multilayer_perceptron.h"
#include "neural/dataset.h"

#include <iostream>

Trainer::Trainer(const TrainingParameters &parameters, const bool &bVerbose) :
    m_rMaxError(parameters.rMaxError),
    m_rCrossValidationEvaluationPercent(parameters.rCrossValidationEvaluationPercent),
    m_bVerbose(bVerbose)
{

}

void Trainer::train(MultilayerPerceptron &multilayerPerceptron, Dataset &dataset) const
{
    // Check activation bounds
    ActivationBounds pfBounds = activationBoundsFromType(multilayerPerceptron.layer(0).activationFunctionType());
    real inputsMin, inputsMax, outputsMin, outputsMax;
    const bool &bBounds = pfBounds(inputsMin, inputsMax, outputsMin, outputsMax);

    if(bBounds == true)
    {
        dataset.normalize(inputsMin, inputsMax, outputsMin, outputsMax);
    }

    const real rDatasetSize = static_cast<real>(dataset.size());
    const size_t crossValidationIndex = static_cast<size_t>(rDatasetSize * m_rCrossValidationEvaluationPercent);
    
    // Compute Evaluation Error
    real rError = 0.0;
    for(size_t i = crossValidationIndex; i < dataset.size(); ++i)
    {
        const Outputs &actualOutputs = multilayerPerceptron.evaluate(dataset.inputs(i));

        // Add euclidian distance between target and actual Output to Error
        for(size_t j = 0; j < actualOutputs.size(); ++j)
        {
            const real &rDifference = actualOutputs[j] - dataset.outputs(i)[j];
            rError += sqrt(rDifference * rDifference);
        }
    }
    rError /= rDatasetSize;

    if(m_bVerbose == true)
    {
        std::cout << "Start training" << std::endl
            << "Training size: " << crossValidationIndex << std::endl
            << "Evaluation size: " << (dataset.size() - crossValidationIndex) << std::endl
            << "Goal Error: " << m_rMaxError << std::endl
            << "Current Error: " << rError << std::endl << std::endl;
    }

    real rPreviousError = m_rMaxError;
    unsigned long iterationsIndex = 0;
    bool bEnd = false;
    while(rError > m_rMaxError && iterationsIndex <= s_maxIterations && bEnd != true)
    {
        // Train Neural Network
        for(size_t i = 0; i < crossValidationIndex; ++i)
        {
            multilayerPerceptron.train(dataset.inputs(i), dataset.outputs(i));
        }

        // Compute Evaluation Error
        rError = 0.0;
        for(size_t i = crossValidationIndex; i < dataset.size(); ++i)
        {
            const Outputs &actualOutputs = multilayerPerceptron.evaluate(dataset.inputs(i));

            // Add euclidian distance between target and actual Output to Error
            for(size_t j = 0; j < actualOutputs.size(); ++j)
            {
                const real &rDifference = actualOutputs[j] - dataset.outputs(i)[j];
                rError += sqrt(rDifference * rDifference);
            }
        }
        rError /= rDatasetSize;

        if(m_bVerbose == true)
        {
            std::cout << "Iteration " << iterationsIndex << std::endl
                << "Goal Error: " << m_rMaxError << std::endl
                << "Current Error: " << rError << std::endl << std::endl;
        }

        if(rError > rPreviousError)
        {
            bEnd = true;
        }

        rPreviousError = rError;

        ++iterationsIndex;
    }

    if(m_bVerbose == true)
    {
        std::cout << "End training" << std::endl
            << "Goal Error: " << m_rMaxError << std::endl
            << "Final Error: " << rError << std::endl << std::endl;
    }

    // TODO denormalize Dataset?
}

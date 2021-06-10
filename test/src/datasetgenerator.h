#ifndef DATASETGENERATOR_H
#define DATASETGENERATOR_H

#include "neural/dataset.h"

#include <chrono>
#include <random>

using GenerationFunctionPtr = void(*)(const LayerInputs &, LayerOutputs &);

namespace DatasetGenerator
{
    Dataset generateRandomDataset(const size_t &datasetSize, const real &rInputLowerBound, const real &rInputUpperBound, const MultilayerPerceptronParameters &networkParameters, GenerationFunctionPtr generationFunctionPtr)
    {
        if(datasetSize > 0 && rInputLowerBound < rInputUpperBound)
        {
            size_t numberOfInputs = networkParameters.numberOfInputs;
            size_t numberOfOutputs = networkParameters.aLayerParameters.back().layerSize;

            std::vector<std::vector<real>> aInputs(datasetSize);
            std::vector<std::vector<real>> aOutputs(datasetSize);

            std::uniform_real_distribution<real> unif(rInputLowerBound, rInputUpperBound);
            std::default_random_engine re(e_uiSeed++);

            for(size_t i = 0; i < datasetSize; ++i)
            {
                aInputs[i].resize(numberOfInputs);
                for(int j = 0; j < numberOfInputs; ++j)
                {
                    aInputs[i][j] = unif(re);
                }
                aOutputs[i].resize(numberOfOutputs);
                for(int j = 0; j < numberOfOutputs; ++j)
                {
                    generationFunctionPtr(aInputs[i], aOutputs[i]);
                }
            }
            return Dataset(aInputs, aOutputs);
        }
        return Dataset();
    }

    Dataset generateSampledDataset(const size_t &datasetSize, const real &rInputLowerBound, const real &rInputUpperBound, const MultilayerPerceptronParameters &networkParameters, GenerationFunctionPtr generationFunctionPtr)
    {
        if(datasetSize > 0 && rInputLowerBound < rInputUpperBound)
        {
            size_t numberOfInputs = networkParameters.numberOfInputs;
            size_t numberOfOutputs = networkParameters.aLayerParameters.back().layerSize;

            std::vector<std::vector<real>> aInputs(datasetSize);
            std::vector<std::vector<real>> aOutputs(datasetSize);

            real rDelta = 0.0;
            real rLowerBound = rInputLowerBound;
            real rUpperBound = rInputUpperBound;

            if(rInputLowerBound < 0)
            {
                rDelta = rInputLowerBound;
                rUpperBound = rInputUpperBound - rInputLowerBound;
                rLowerBound = 0.0;
            }

            real rStep = (rUpperBound - rLowerBound) / static_cast<real>(datasetSize);

            real rInputValue = rLowerBound;
            for(size_t i = 0; i < datasetSize; ++i)
            {
                aInputs[i].resize(numberOfInputs);
                for(int j = 0; j < numberOfInputs; ++j)
                {
                    aInputs[i][j] = rInputValue + rDelta;
                    rInputValue += rStep;
                }
                aOutputs[i].resize(numberOfOutputs);
                for(int j = 0; j < numberOfOutputs; ++j)
                {
                    generationFunctionPtr(aInputs[i], aOutputs[i]);
                }
            }
            return Dataset(aInputs, aOutputs);
        }
        return Dataset();
    }
};

#endif // DATASETGENERATOR_H

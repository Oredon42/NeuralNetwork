#ifndef DATASETGENERATOR_H
#define DATASETGENERATOR_H

#include "src/defines.h"
#include <chrono>
#include <random>

namespace DatasetGenerator
{
    Dataset generateRandomDataset(const std::size_t &datasetSize, const real &rInputLowerBound, const real &rInputUpperBound, Outputs (*outputFunction)(Inputs))
    {
        if(datasetSize > 0 && rInputLowerBound < rInputUpperBound)
        {
            std::vector<Inputs> aInputs(datasetSize);
            std::vector<Outputs> aOutputs(datasetSize);

            real rDelta = 0.0;
            real rLowerBound = rInputLowerBound;
            real rUpperBound = rInputUpperBound;

            if(rInputLowerBound < 0)
            {
                rDelta = rInputLowerBound;
                rUpperBound = rInputUpperBound - rInputLowerBound;
                rLowerBound = 0.0;
            }

            std::uniform_real_distribution<real> unif(rLowerBound, rUpperBound);
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine re(seed);

            for(std::size_t i = 0; i < datasetSize; ++i)
            {
                aInputs[i].push_back(unif(re) + rDelta);
                aOutputs[i] = outputFunction(aInputs[i]);
            }
            return {aInputs, aOutputs};
        }
        return {std::vector<Inputs>(), std::vector<Outputs>()};
    }

    Dataset generateSampledDataset(const std::size_t &datasetSize, const real &rInputLowerBound, const real &rInputUpperBound, Outputs (*outputFunction)(Inputs))
    {
        if(datasetSize > 0 && rInputLowerBound < rInputUpperBound)
        {
            std::vector<Inputs> aInputs(datasetSize);
            std::vector<Outputs> aOutputs(datasetSize);

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
            for(real i = 0; i < datasetSize; ++i)
            {
                aInputs[i].push_back(rInputValue + rDelta);
                aOutputs[i] = outputFunction(aInputs[i]);
                rInputValue += rStep;
            }
            return {aInputs, aOutputs};
        }
        return {std::vector<Inputs>(), std::vector<Outputs>()};
    }
};

#endif // DATASETGENERATOR_H
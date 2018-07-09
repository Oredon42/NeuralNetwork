#ifndef DATASETGENERATOR_H
#define DATASETGENERATOR_H

#include "neural/dataset.h"

#include <chrono>
#include <random>

namespace DatasetGenerator
{
    Dataset generateRandomDataset(const size_t &datasetSize, const double &rInputLowerBound, const double &rInputUpperBound, std::vector<double>(*outputFunction)(const std::vector<double> &))
    {
        if(datasetSize > 0 && rInputLowerBound < rInputUpperBound)
        {
			std::vector<std::vector<double>> aInputs(datasetSize);
			std::vector<std::vector<double>> aOutputs(datasetSize);

            std::uniform_real_distribution<double> unif(rInputLowerBound, rInputUpperBound);
            const unsigned int &uiSeed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine re(uiSeed);

            for(size_t i = 0; i < datasetSize; ++i)
            {
				aInputs[i].resize(2);
                aInputs[i][0] = unif(re);
				aInputs[i][1] = unif(re);
				aOutputs[i].resize(1);
                aOutputs[i][0] = outputFunction(aInputs[i])[0];
            }
            return Dataset(aInputs, aOutputs);
        }
        return Dataset();
    }

    Dataset generateSampledDataset(const size_t &datasetSize, const double &rInputLowerBound, const double &rInputUpperBound, std::vector<double>(*outputFunction)(const std::vector<double> &))
    {
        if(datasetSize > 0 && rInputLowerBound < rInputUpperBound)
        {
			std::vector<std::vector<double>> aInputs(datasetSize);
			std::vector<std::vector<double>> aOutputs(datasetSize);

            double rDelta = 0.0;
			double rLowerBound = rInputLowerBound;
			double rUpperBound = rInputUpperBound;

            if(rInputLowerBound < 0)
            {
                rDelta = rInputLowerBound;
                rUpperBound = rInputUpperBound - rInputLowerBound;
                rLowerBound = 0.0;
            }

			double rStep = (rUpperBound - rLowerBound) / static_cast<double>(datasetSize);

			double rInputValue = rLowerBound;
            for(size_t i = 0; i < datasetSize; ++i)
            {
				aInputs[i].resize(2);
				aInputs[i][0] = rInputValue + rDelta;
                aOutputs[i] = outputFunction(aInputs[i]);
                rInputValue += rStep;
            }
            return Dataset(aInputs, aOutputs);
        }
        return Dataset();
    }
};

#endif // DATASETGENERATOR_H
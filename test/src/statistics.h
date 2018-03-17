#ifndef STATISTICS_H
#define STATISTICS_H

#include "src/defines.h"
#include <iostream>
#include <math.h>

namespace Statistics
{
    void printStatistics(const std::vector<Outputs> &aOutputs, const std::vector<Outputs> &aWantedOutputs)
    {
        if(aOutputs.size() > 0 && aOutputs.size() == aWantedOutputs.size())
        {
			std::vector<real> rErrorSum(aOutputs.size(), 0.0);

            for(size_t i = 0; i < aOutputs.size(); ++i)
            {
				if(aOutputs[i].size() > 0 && aOutputs[i].size() == aWantedOutputs[i].size())
				{
					for(size_t j = 0; j < aOutputs[i].size(); ++j)
					{
						rErrorSum[j] += fabs(aOutputs[i][j] - aWantedOutputs[i][j]);
					}
				}
            }

            std::cout << "Nb data: " << aOutputs.size() << std::endl;
			for(size_t i = 0; i < aOutputs[0].size(); ++i)
			{
				std::cout << "Error sum " << i << ": " << rErrorSum[i] << std::endl;
				std::cout << "Error mean " << i << ": " << ( rErrorSum[i] / aOutputs.size() ) << std::endl;
			}
        }
    }
};

#endif // STATISTICS_H
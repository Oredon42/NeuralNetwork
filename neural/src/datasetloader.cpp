#include "datasetloader.h"

#include "defines.h"
#include <fstream>

Dataset DatasetLoader::load(const std::string &strDatasetPath)
{
	std::vector<Inputs> aInputs;
	std::vector<Outputs> aOutputs;

	std::ifstream loadFile;
	loadFile.open(strDatasetPath);

	if(loadFile.is_open() == true)
	{
		size_t inputsSize, outputsSize;
		if(loadFile >> inputsSize >> outputsSize)
		{
			bool bLoadingFinished = true;
			while(bLoadingFinished == true)
			{
				Inputs inputs(inputsSize);
				for(size_t i = 0; i < inputsSize; ++i)
				{
					real rValue;
					if(!(loadFile >> rValue))
					{
						bLoadingFinished = false;
						break;
					}
					inputs.push_back(rValue);
				}
				aInputs.push_back(inputs);
				Outputs outputs(outputsSize);
				for(size_t i = 0; i < outputsSize; ++i)
				{
					real rValue;
					if(!(loadFile >> rValue))
					{
						bLoadingFinished = false;
						break;
					}
					outputs.push_back(rValue);
				}
				aOutputs.push_back(outputs);
			}
		}
		loadFile.close();
	}

	return {aInputs, aOutputs};
}

void DatasetLoader::save(const Dataset &dataset, const std::string &strDatasetPath)
{
	if(dataset.aInputs.size() > 0 || dataset.aOutputs.size() > 0)
	{
		std::ofstream saveFile;
		saveFile.open(strDatasetPath);

		if(saveFile.is_open() == true)
		{
			size_t	inputsSize = (dataset.aInputs.size() > 0) ?  dataset.aInputs[0].size() : 0,
					outputsSize = (dataset.aOutputs.size() > 0) ? dataset.aOutputs[0].size() : 0;
			saveFile << inputsSize << outputsSize;

			size_t maxIndex = (inputsSize > outputsSize) ? inputsSize : outputsSize;
			for(size_t i = 0; i < maxIndex; ++i)
			{
				if(dataset.aInputs.size() > i)
				{
					for (size_t j = 0; j < inputsSize; ++j)
					{
						saveFile << dataset.aInputs[i][j];
					}
				}
				if(dataset.aOutputs.size() > i)
				{
					for(size_t j = 0; j < outputsSize; ++j)
					{
						saveFile << dataset.aOutputs[i][j];
					}
				}
			}
		}
		saveFile.close();
	}
}

#include "neural/dataset.h"

#include "neural/assert.h"

#include <fstream>

Dataset::Dataset()
{
}

Dataset::Dataset(const std::vector<Inputs> &aInputs, const std::vector<Outputs> &aOutputs, const DatasetParameters &parameters)
{
	ASSERT(aInputs.size() == aOutputs.size());

	m_aInputs = aInputs;
	m_aOutputs = aOutputs;

	if(parameters.inputsMin.size() > 0)
	{
		m_inputsMin = parameters.inputsMin;
		m_inputsMax = parameters.inputsMax;
		m_outputsMin = parameters.outputsMin;
		m_outputsMax = parameters.outputsMax;
	}
	else
	{
		computeMinMax();
	}
}

void Dataset::normalize(const real &rInputsMin, const real &rInputsMax, const real &rOutputsMin, const real &rOutputsMax)
{
	ASSERT(size() > 0);

	const size_t &inputSize = m_aInputs[0].size();
	const size_t &outputSize = m_aOutputs[0].size();

	// Precompute normalization data
	Outputs aDenormalizedInputsDiff(inputSize);
	const real aNormalizedInputsDiff = rInputsMax - rInputsMin;
	Outputs aDenormalizedOutputsDiff(inputSize);
	const real aNormalizedOutputsDiff = rOutputsMax - rOutputsMin;

	for(size_t j = 0; j < inputSize; ++j)
	{
		aDenormalizedInputsDiff[j] = 1.0 / (m_inputsMax[j] - m_inputsMin[j]);
	}
	for(size_t j = 0; j < outputSize; ++j)
	{
		aDenormalizedOutputsDiff[j] = 1.0 / (m_outputsMax[j] - m_outputsMin[j]);
	}

	// Normalize
	for(size_t i = 0; i < size(); ++i)
	{
		for(size_t j = 0; j < inputSize; ++j)
		{
			m_aInputs[i][j] = (m_aInputs[i][j] - m_inputsMin[j]) * aDenormalizedInputsDiff[j] * aNormalizedInputsDiff + rInputsMin;
		}
		for(size_t j = 0; j < outputSize; ++j)
		{
			m_aOutputs[i][j] = (m_aOutputs[i][j] - m_outputsMin[j]) * aDenormalizedOutputsDiff[j] * aNormalizedOutputsDiff + rOutputsMin;
		}
	}
}

void Dataset::addData(const Inputs &aInputs, const Outputs &aOutputs)
{
	ASSERT(m_aInputs.size() == 0 || (m_aInputs.size() == m_aInputs[0].size() && m_aOutputs[0].size() == aOutputs.size()));

	m_aInputs.push_back(aInputs);
	m_aOutputs.push_back(aOutputs);

	// Update min/max
	for(size_t i = 0; i < aInputs.size(); ++i)
	{
		if(aInputs[i] < m_inputsMin[i])
		{
			m_inputsMin[i] = aInputs[i];
		}
		else if(aInputs[i] > m_inputsMax[i])
		{
			m_inputsMax[i] = aInputs[i];
		}
	}
	for(size_t i = 0; i < aOutputs.size(); ++i)
	{
		if(aOutputs[i] < m_outputsMin[i])
		{
			m_outputsMin[i] = aOutputs[i];
		}
		else if(aOutputs[i] > m_outputsMax[i])
		{
			m_outputsMax[i] = aOutputs[i];
		}
	}
}

void Dataset::loadFile(const std::string &strDatasetPath)
{
	m_aInputs.clear();
	m_aOutputs.clear();

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
				m_aInputs.push_back(inputs);

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
				m_aOutputs.push_back(outputs);
			}
		}
		loadFile.close();
	}

	computeMinMax();
}

void Dataset::writeFile(const std::string &strDatasetPath) const
{
	if(m_aInputs.size() > 0 || m_aOutputs.size() > 0)
	{
		std::ofstream saveFile;
		saveFile.open(strDatasetPath);

		if(saveFile.is_open() == true)
		{
			size_t	inputsSize = (m_aInputs.size() > 0) ? m_aInputs[0].size() : 0,
					outputsSize = (m_aOutputs.size() > 0) ? m_aOutputs[0].size() : 0;
			saveFile << inputsSize << outputsSize;

			size_t maxIndex = (inputsSize > outputsSize) ? inputsSize : outputsSize;
			for(size_t i = 0; i < maxIndex; ++i)
			{
				if(m_aInputs.size() > i)
				{
					for(size_t j = 0; j < inputsSize; ++j)
					{
						saveFile << m_aInputs[i][j];
					}
				}
				if(m_aOutputs.size() > i)
				{
					for(size_t j = 0; j < outputsSize; ++j)
					{
						saveFile << m_aOutputs[i][j];
					}
				}
			}
		}
		saveFile.close();
	}
}

void Dataset::computeMinMax()
{
	m_inputsMin = m_aInputs[0];
	m_inputsMax = m_inputsMin;
	m_outputsMin = m_aOutputs[0];
	m_outputsMax = m_outputsMin;

	const size_t inputSize = m_aInputs[0].size();
	const size_t outputSize = m_aOutputs[0].size();

	// Get min and max of all Inputs/Outputs
	for(size_t i = 1; i < size(); ++i)
	{
		// Inputs
		for(size_t j = 0; j < inputSize; ++j)
		{
			if(m_aInputs[i][j] < m_inputsMin[j])
			{
				m_inputsMin[j] = m_aInputs[i][j];
			}
			else if(m_aInputs[i][j] > m_inputsMax[j])
			{
				m_inputsMax[j] = m_aInputs[i][j];
			}
		}
		// Outputs
		for(size_t j = 0; j < outputSize; ++j)
		{
			if(m_aOutputs[i][j] < m_outputsMin[j])
			{
				m_outputsMin[j] = m_aOutputs[i][j];
			}
			else if(m_aOutputs[i][j] > m_outputsMax[j])
			{
				m_outputsMax[j] = m_aOutputs[i][j];
			}
		}
	}
}

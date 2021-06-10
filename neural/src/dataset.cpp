#include "neural/dataset.h"

#include "neural/assert.h"

#include <fstream>

Dataset::Dataset()
{
}

Dataset::Dataset(const std::vector<LayerInputs> &aInputs, const std::vector<LayerOutputs> &aOutputs, const DatasetParameters &parameters)
{
    ASSERT(aInputs.size() == aOutputs.size());

    m_aInputs = aInputs;
    m_aOutputs = aOutputs;

    if(parameters.filled() == true)
    {
        m_inputsMin = parameters.inputsMin;
        m_inputsMax = parameters.inputsMax;
        m_outputsMin = parameters.outputsMin;
        m_outputsMax = parameters.outputsMax;

        m_inputsMean = parameters.inputsMean;
        m_inputsStandardDeviation = parameters.inputsStandardDeviation;
        m_outputsMean = parameters.outputsMean;
        m_outputsStandardDeviation = parameters.outputsStandardDeviation;
    }
    else
    {
        computeStatistics();
    }
}

void Dataset::normalise()
{
    ASSERT(size() > 0);

    real rInputsMin = 0.0;
    real rInputsMax = 1.0;
    real rOutputsMin = 0.0;
    real rOutputsMax = 1.0;

    const size_t inputSize = m_aInputs[0].size();
    const size_t outputSize = m_aOutputs[0].size();

    // Precompute normalization data
    LayerOutputs aDenormalizedInputsDiff(inputSize);
    const real aNormalizedInputsDiff = rInputsMax - rInputsMin;
    LayerOutputs aDenormalizedOutputsDiff(outputSize);
    const real aNormalizedOutputsDiff = rOutputsMax - rOutputsMin;

    for(size_t i = 0; i < inputSize; ++i)
    {
        aDenormalizedInputsDiff[i] = 1.0 / (m_inputsMax[i] - m_inputsMin[i]);
    }
    for(size_t i = 0; i < outputSize; ++i)
    {
        aDenormalizedOutputsDiff[i] = 1.0 / (m_outputsMax[i] - m_outputsMin[i]);
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

void Dataset::standardise()
{
    const size_t inputSize = m_aInputs[0].size();
    const size_t outputSize = m_aOutputs[0].size();

    for(size_t i = 0; i < size(); ++i)
    {
        for(size_t j = 0; j < inputSize; ++j)
        {
            m_aInputs[i][j] = (m_aInputs[i][j] - m_inputsMean[j]) / m_inputsStandardDeviation[j];
        }
        for(size_t j = 0; j < outputSize; ++j)
        {
            m_aOutputs[i][j] = (m_aOutputs[i][j] - m_outputsMean[j]) / m_outputsStandardDeviation[j];
        }
    }
}

void Dataset::addData(const LayerInputs &aInputs, const LayerOutputs &aOutputs)
{
    ASSERT(m_aInputs.size() == 0 || (m_aInputs[0].size() == aInputs.size() && m_aOutputs[0].size() == aOutputs.size()));

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
                LayerInputs inputs(inputsSize);
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

                LayerOutputs outputs(outputsSize);
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

    computeStatistics();
}

void Dataset::writeFile(const std::string &strDatasetPath) const
{
    if(m_aInputs.size() > 0 || m_aOutputs.size() > 0)
    {
        std::ofstream saveFile;
        saveFile.open(strDatasetPath);

        if(saveFile.is_open() == true)
        {
            size_t    inputsSize = (m_aInputs.size() > 0) ? m_aInputs[0].size() : 0,
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

void Dataset::computeStatistics()
{
    m_inputsMin = m_aInputs[0];
    m_inputsMax = m_inputsMin;
    m_outputsMin = m_aOutputs[0];
    m_outputsMax = m_outputsMin;

    const size_t inputSize = m_aInputs[0].size();
    const size_t outputSize = m_aOutputs[0].size();

    m_inputsMean = m_aInputs[0];
    m_outputsMean = m_aOutputs[0];
    m_inputsStandardDeviation = LayerInputs(inputSize, 0.0);
    m_outputsStandardDeviation = LayerInputs(outputSize, 0.0);

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

            m_inputsMean[j] += m_aInputs[i][j];
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

            m_outputsMean[j] += m_aOutputs[i][j];
        }
    }

    for(size_t j = 0; j < inputSize; ++j)
    {
        m_inputsMean[j] /= size();
    }

    for(size_t j = 0; j < outputSize; ++j)
    {
        m_outputsMean[j] /= size();
    }

    for(size_t i = 0; i < size(); ++i)
    {
        for(size_t j = 0; j < inputSize; ++j)
        {
            real rDeviation = m_aInputs[i][j] - m_inputsMean[j];
            m_inputsStandardDeviation[j] += rDeviation * rDeviation;
        }

        for(size_t j = 0; j < outputSize; ++j)
        {
            real rDeviation = m_aOutputs[i][j] - m_outputsMean[j];
            m_outputsStandardDeviation[j] += rDeviation * rDeviation;
        }
    }

    for(size_t j = 0; j < inputSize; ++j)
    {
        m_inputsStandardDeviation[j] = sqrt(m_inputsStandardDeviation[j] / size());
    }

    for(size_t j = 0; j < outputSize; ++j)
    {
        m_outputsStandardDeviation[j] = sqrt(m_outputsStandardDeviation[j] / size());
    }
}

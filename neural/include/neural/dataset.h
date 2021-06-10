#ifndef DATASET_H
#define DATASET_H

#include <string>
#include "neural/defines.h"

struct DatasetParameters
{
    LayerInputs inputsMin;
    LayerInputs inputsMax;
    LayerOutputs outputsMin;
    LayerOutputs outputsMax;

    LayerInputs inputsMean;
    LayerInputs inputsStandardDeviation;
    LayerOutputs outputsMean;
    LayerOutputs outputsStandardDeviation;

    bool filled() const
    {
        return !inputsMin.empty() && !inputsMax.empty() && !outputsMin.empty() &&
            !outputsMax.empty() && !inputsMean.empty() && !inputsStandardDeviation.empty() &&
            !outputsMean.empty() && !outputsStandardDeviation.empty();
    }
};

class Dataset
{
public:
    Dataset();
    Dataset(const std::vector<LayerInputs> &aInputs, const std::vector<LayerOutputs> &aOutputs, const DatasetParameters &parameters = DatasetParameters());

    INLINE const LayerInputs &inputs(const size_t i) const{return m_aInputs[i];}
    INLINE const LayerOutputs &outputs(const size_t i) const{return m_aOutputs[i];}
    INLINE size_t size() const{return m_aInputs.size();}

    void normalise();
    void standardise();

    void addData(const LayerInputs &aInputs, const LayerOutputs &aOutputs);

    void loadFile(const std::string &strDatasetPath);
    void writeFile(const std::string &strDatasetPath) const;

private:
    std::vector<LayerInputs> m_aInputs;
    std::vector<LayerOutputs> m_aOutputs;

    LayerInputs m_inputsMin;
    LayerInputs m_inputsMax;
    LayerOutputs m_outputsMin;
    LayerOutputs m_outputsMax;

    LayerInputs m_inputsMean;
    LayerInputs m_inputsStandardDeviation;
    LayerOutputs m_outputsMean;
    LayerOutputs m_outputsStandardDeviation;
    
private:
    void computeStatistics();
};

#endif // DATASET_H

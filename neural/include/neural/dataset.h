#ifndef DATASET_H
#define DATASET_H

#include <string>
#include "neural/defines.h"

class Dataset
{
public:
    struct DatasetParameters
    {
        Inputs inputsMin;
        Inputs inputsMax;
        Outputs outputsMin;
        Outputs outputsMax;
    };

public:
    Dataset();
    Dataset(const std::vector<Inputs> &aInputs, const std::vector<Outputs> &aOutputs, const DatasetParameters &parameters = DatasetParameters());

    INLINE Inputs inputs(const size_t i) const{return m_aInputs[i];}
    INLINE Outputs outputs(const size_t i) const{return m_aOutputs[i];}
    INLINE size_t size() const{return m_aInputs.size();}
    INLINE Inputs inputsMin() const{return m_inputsMin;}
    INLINE Inputs inputsMax() const{return m_inputsMax;}
    INLINE Outputs outputsMin() const{return m_outputsMin;}
    INLINE Outputs outputsMax() const{return m_outputsMax;}

    void normalize(const real &inputsMin = -1.0, const real &inputsMax = 1.0, const real &outputsMin = -1.0, const real &outputsMax = 1.0);
    void addData(const Inputs &aInputs, const Outputs &aOutputs);

    void loadFile(const std::string &strDatasetPath);
    void writeFile(const std::string &strDatasetPath) const;

private:
    std::vector<Inputs> m_aInputs;
    std::vector<Outputs> m_aOutputs;

    Inputs m_inputsMin;
    Inputs m_inputsMax;
    Outputs m_outputsMin;
    Outputs m_outputsMax;
    
private:
    void computeMinMax();
};

#endif // DATASET_H

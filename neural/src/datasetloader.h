#ifndef DATASETLOADER_H
#define DATASETLOADER_H

#include <string>
struct Dataset;

namespace DatasetLoader
{
	Dataset load(const std::string &strDatasetPath);
	void save(const Dataset &dataset, const std::string &strDatasetPath);
};

#endif // DATASETLOADER_H
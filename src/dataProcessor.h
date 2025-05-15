#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <vector>
#include <string> 
using namespace std;

class dataProcessor {
private:
    std::vector<float> globalMedians;
    std::vector<float> healthyMedians;
    std::vector<float> atRiskMedians;

    std::string classifyUserHealth(const std::vector<float>& rawData); // New method

public:
    dataProcessor(const std::vector<std::vector<float>>& data, const std::vector<std::string>& labels);

    std::vector<float> cleanData(const std::vector<float>& rawData);
    std::vector<float> normalizeData(const std::vector<float>& cleanedData);
};
#endif
#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <vector>

class dataProcessor {
public:
    dataProcessor();
    std::vector<float> cleanData(std::vector<float>& rawData);
    std::vector<float> normalizeData(std::vector<float>& cleanedData);
};

#endif
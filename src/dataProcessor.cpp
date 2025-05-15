#include "dataProcessor.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <float.h>  // for FLT_MAX

using namespace std;


vector<float> computeMedians(const vector<vector<float>>& data) { // Helper to compute median for each column
    if (data.empty()) return {};

    size_t numFeatures = data[0].size();
    vector<float> medians(numFeatures);

    for (size_t i = 0; i < numFeatures; ++i) {
        vector<float> column;
        for (const auto& row : data) {
            column.push_back(row[i]);
        }

        sort(column.begin(), column.end());
        if (column.size() % 2 == 0)
            medians[i] = (column[column.size() / 2 - 1] + column[column.size() / 2]) / 2.0f;
        else
            medians[i] = column[column.size() / 2];
    }

    return medians;
}

dataProcessor::dataProcessor(const vector<vector<float>>& data, const vector<string>& labels) {
    vector<vector<float>> healthy, atRisk;
    for (size_t i = 0; i < data.size(); ++i) {
        if (labels[i] == "Not At Risk") healthy.push_back(data[i]);
        else atRisk.push_back(data[i]);
    }

    globalMedians = computeMedians(data);
    healthyMedians = computeMedians(healthy);
    atRiskMedians = computeMedians(atRisk);
}

vector<float> dataProcessor::cleanData(const vector<float>& rawData) {
    vector<float> cleaned = rawData;

    for (size_t i = 0; i < cleaned.size(); ++i) {
        if (cleaned[i] == -1.0f) {
            cleaned[i] = globalMedians[i];
        }
    }

    return cleaned;
}


vector<float> dataProcessor::normalizeData(const vector<float>& cleanedData) {
    vector<float> normalized = cleanedData; 
    vector<float> minVals = {0, 0, 0.0, 0.0, 0.0, 0.0, 0, 40.0, 0};  
    vector<float> maxVals = {1, 100, 12.0, 10.0, 100.0, 10.0, 2, 130.0, 25000};  

    for (size_t i = 0; i < normalized.size(); ++i) {
        normalized[i] = (normalized[i] - minVals[i]) / (maxVals[i] - minVals[i]);
    }

    return normalized;
}
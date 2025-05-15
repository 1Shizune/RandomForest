#include "dataProcessor.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <float.h>
#include <iostream>

using namespace std;

const vector<float> dataProcessor::MIN_VALS = {
    0,    // gender
    0,    // age
    0.0,  // sleep duration
    0.0,  // sleep quality
    10.0, // physical activity
    0.0,  // stress level
    0,    // BMI
    30.0, // heart rate
    1000  // steps
};

const vector<float> dataProcessor::MAX_VALS = {
    1,     
    120,   
    24.0,  
    10.0,  
    100.0, 
    10.0,  
    2,     
    200.0, 
    25000  
};

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

string dataProcessor::classifyUserHealth(const vector<float>& rawData) {
    int score = 0;

    // Use available values only
    if (rawData[3] != -1.0f && rawData[3] >= 6.0f) score++;  // Quality of sleep
    if (rawData[4] != -1.0f && rawData[4] >= 30.0f) score++; // Physical activity
    if (rawData[5] != -1.0f && rawData[5] <= 4.0f) score++;  // Low stress

    if (rawData[3] != -1.0f && rawData[3] < 4.0f) score--;   // Poor sleep quality
    if (rawData[4] != -1.0f && rawData[4] < 20.0f) score--;  // Sedentary
    if (rawData[5] != -1.0f && rawData[5] >= 7.0f) score--;  // High stress

    if (score >= 2) return "Healthy";
    else if (score <= -2) return "AtRisk";
    else return "Neutral";
}

vector<float> dataProcessor::cleanData(const vector<float>& rawData) {
    // Step 1: Clean raw data with global medians
    vector<float> cleaned = cleanData(rawData, "global");

    // Step 2: Classify user based on the globally cleaned version
    string classification = classifyUserHealth(cleaned);

    // Optional: Debug classification and input
    cout << "Initial classification (based on global cleaned): " << classification << endl;

    // Step 3: Re-clean original raw input using group-specific medians
    if (classification == "Healthy") {
        return cleanData(rawData, "healthy");
    } else {
        return cleanData(rawData, "atRisk");
    }
}



vector<float> dataProcessor::cleanData(const vector<float>& rawData, const string& strategy) { //Clean data by replacing -1.0f with corresponding medians
    vector<float> cleaned = rawData;

    const vector<float>* medians;
    if (strategy == "global") {
        medians = &globalMedians;
    } else if (strategy == "healthy") {
        medians = &healthyMedians;
    } else if (strategy == "atRisk") {
        medians = &atRiskMedians;
    } else {
        throw invalid_argument("Unknown imputation strategy: " + strategy);
    }

    for (size_t i = 0; i < cleaned.size(); ++i) {
        if (cleaned[i] == -1.0f || cleaned[i] < MIN_VALS[i] || cleaned[i] > MAX_VALS[i]) {
            cleaned[i] = (*medians)[i];
        }
    }

        return cleaned;
}

// Normalize data to 0-1 range using fixed feature bounds
vector<float> dataProcessor::normalizeData(const vector<float>& cleanedData) {
    vector<float> normalized = cleanedData;
    vector<float> minVals = MIN_VALS;  // The minimum value for an atribute that can make sense
    vector<float> maxVals = MAX_VALS; // The maximum value for an atribute that can make sense

    for (size_t i = 0; i < normalized.size(); ++i) {
        normalized[i] = (normalized[i] - minVals[i]) / (maxVals[i] - minVals[i]);
    }

    return normalized;
}

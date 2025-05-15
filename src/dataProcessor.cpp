#include "dataProcessor.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <float.h>
#include <iostream>

using namespace std;

// Helper to compute median for each column
vector<float> computeMedians(const vector<vector<float>>& data) {
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

// Constructor
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

// Helper to classify user health based on non-missing fields
string dataProcessor::classifyUserHealth(const vector<float>& rawData) {
    int score = 0;

    // Use available values only
    if (rawData[3] != -1.0f && rawData[3] >= 6.0f) score++;       // Quality of sleep
    if (rawData[4] != -1.0f && rawData[4] >= 30.0f) score++;      // Physical activity
    if (rawData[5] != -1.0f && rawData[5] <= 4.0f) score++;       // Low stress

    if (rawData[3] != -1.0f && rawData[3] < 4.0f) score--;        // Poor sleep quality
    if (rawData[4] != -1.0f && rawData[4] < 20.0f) score--;       // Sedentary
    if (rawData[5] != -1.0f && rawData[5] >= 7.0f) score--;       // High stress

    if (score >= 2) return "Healthy";
    else if (score <= -2) return "AtRisk";
    else return "Neutral";
}

// Clean data by replacing -1.0f with corresponding medians
vector<float> dataProcessor::cleanData(const vector<float>& rawData) {
    vector<float> cleaned = rawData;

    string healthStatus = classifyUserHealth(rawData);
    const vector<float>* sourceMedians;

    if (healthStatus == "Healthy") {
        sourceMedians = &healthyMedians;
        cout << "🟢 User classified as Healthy, using healthy medians for imputation.\n";
    } else if (healthStatus == "AtRisk") {
        sourceMedians = &atRiskMedians;
        cout << "🔴 User classified as At Risk, using at-risk medians for imputation.\n";
    } else {
        sourceMedians = &globalMedians;
        cout << "⚪️ User classified as Neutral, using global medians for imputation.\n";
    }

    for (size_t i = 0; i < cleaned.size(); ++i) {
        if (cleaned[i] == -1.0f) {
            cout << "   ➤ Missing value at index " << i
                 << " replaced with " << (*sourceMedians)[i] << endl;
            cleaned[i] = (*sourceMedians)[i];
        }
    }

    return cleaned;
}

// Normalize data to 0-1 range using fixed feature bounds
vector<float> dataProcessor::normalizeData(const vector<float>& cleanedData) {
    vector<float> normalized = cleanedData;
    vector<float> minVals = {0, 0, 0.0, 0.0, 0.0, 0.0, 0, 40.0, 0};        // feature-wise mins
    vector<float> maxVals = {1, 100, 12.0, 10.0, 100.0, 10.0, 2, 130.0, 25000}; // feature-wise maxes

    for (size_t i = 0; i < normalized.size(); ++i) {
        normalized[i] = (normalized[i] - minVals[i]) / (maxVals[i] - minVals[i]);
    }

    return normalized;
}

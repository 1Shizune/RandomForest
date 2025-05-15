#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <vector>
#include <string>  // ✅ Use lowercase "string", not <String>
using namespace std;

class dataProcessor {
private:
    vector<float> globalMedians;
    vector<float> healthyMedians;
    vector<float> atRiskMedians;

public:
    dataProcessor(const vector<vector<float>>& data, const vector<string>& labels);

    // ✅ Overload for training (with label context)
    vector<float> cleanData(vector<float>& rawData, const string& label);

    // ✅ Overload for prediction (no label)
    vector<float> cleanData(const vector<float>& rawData);

    vector<float> normalizeData(const vector<float>& cleanedData);
};

#endif
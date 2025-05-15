#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <vector>
#include <string> 
using namespace std;

class dataProcessor {
public:
    dataProcessor(const vector<vector<float>>& data, const vector<string>& labels);
    
    vector<float> cleanData(const vector<float>& rawData);
    vector<float> cleanData(const vector<float>& rawData, const string& strategy);
    vector<float> normalizeData(const vector<float>& cleanedData);
    string classifyUserHealth(const vector<float>& rawData);
    
    
    vector<tuple<int, float, string>> getLastReplacements() const; //Expose replacements for logging

private:
    vector<float> globalMedians;
    vector<float> healthyMedians;
    vector<float> atRiskMedians;

    
    vector<tuple<int, float, string>> lastReplacements; //Logging what values were replaced

    static const vector<float> MIN_VALS;
    static const vector<float> MAX_VALS;
};

#endif
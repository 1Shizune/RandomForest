#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <vector>
#include <string>
#include "decisionTree.h"  // Include decisionTree's definition here

using namespace std;

class predictionModel {
private:
    vector<decisionTree> forest;  // Now uses the decisionTree from decisionTree.h
    int nTrees;

public:
    predictionModel(int numTrees = 10);
    void trainModel(const vector<vector<float>>& data, const vector<string>& labels);
    string predict(const vector<float>& inputData);
};

#endif
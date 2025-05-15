#ifndef PREDICTIONMODEL_H
#define PREDICTIONMODEL_H

#include <vector>
#include <string>
#include "decisionTree.h" 
using namespace std;

class predictionModel {
private:
    vector<decisionTree> forest;
    int nTrees;

public:
    predictionModel(int numTrees = 10);
    void trainModel(const vector<vector<float>>& data, const vector<string>& labels);
    string predict(const vector<float>& inputData);
};

#endif

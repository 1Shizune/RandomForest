#ifndef PREDICTIONMODEL_H
#define PREDICTIONMODEL_H

#include <vector>
#include <string>
#include "decisionTree.h"  // Include decisionTree for model training

class predictionModel {
private:
    std::vector<decisionTree> forest;
    int nTrees;

public:
    predictionModel(int numTrees = 10);
    void trainModel(const std::vector<std::vector<float>>& data, const std::vector<std::string>& labels);
    std::string predict(const std::vector<float>& inputData);
};

#endif

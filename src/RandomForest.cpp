#include "RandomForest.h"
#include "decisionTree.h"  // Include the decision tree header
#include <map>
#include <cstdlib>
#include <ctime>
#include <vector>

// Remove the duplicate constructor definition from here

predictionModel::predictionModel(int numTrees) : nTrees(numTrees) {}

void predictionModel::trainModel(const std::vector<std::vector<float>>& data, 
                               const std::vector<std::string>& labels) {
    srand(time(0));
    for (int i = 0; i < nTrees; ++i) {
        std::vector<std::vector<float>> sampleData;
        std::vector<std::string> sampleLabels;

        // Bootstrap sampling
        for (size_t j = 0; j < data.size(); ++j) {
            int idx = rand() % data.size();
            sampleData.push_back(data[idx]);
            sampleLabels.push_back(labels[idx]);
        }

        decisionTree tree(5, 2);
        tree.train(sampleData, sampleLabels);
        forest.push_back(tree);
    }
}

std::string predictionModel::predict(const std::vector<float>& inputData) {
    std::map<std::string, int> voteCount;
    for (auto& tree : forest) {
        std::string vote = tree.predict(inputData);
        voteCount[vote]++;
    }

    std::string finalPrediction;
    int maxVotes = 0;
    for (auto& [label, count] : voteCount) {
        if (count > maxVotes) {
            finalPrediction = label;
            maxVotes = count;
        }
    }
    return finalPrediction;
}
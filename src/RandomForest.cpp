#include "RandomForest.h"
#include "decisionTree.h"  // Include the decision tree header
#include <map>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

predictionModel::predictionModel(int numTrees) : nTrees(numTrees) {}

void predictionModel::trainModel(const vector<vector<float>>& data, 
    const vector<string>& labels) {
    srand(time(0));
    for (int i = 0; i < nTrees; ++i) {
        vector<vector<float>> sampleData;
        vector<string> sampleLabels;

        
        for (size_t j = 0; j < data.size(); ++j) { //Bootstrap sampling
            int idx = rand() % data.size();
            sampleData.push_back(data[idx]);
            sampleLabels.push_back(labels[idx]);
        }

        decisionTree tree(5, 2);
        tree.train(sampleData, sampleLabels);
        forest.push_back(tree);
    }
}

string predictionModel::predict(const vector<float>& inputData) {
    map<string, int> voteCount;
    for (auto& tree : forest) {
        string vote = tree.predict(inputData);
        voteCount[vote]++;
    }

    string finalPrediction;
    int maxVotes = 0;
    for (auto& [label, count] : voteCount) {
        if (count > maxVotes) {
            finalPrediction = label;
            maxVotes = count;
        }
    }
    return finalPrediction;
}
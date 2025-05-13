#pragma once
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <unordered_map>
using namespace std;

class DecisionTree {
public:
    DecisionTree(int maxDepth = 5, int minSamples = 2);
    void train(const vector<vector<float>>& data, const vector<string>& labels);
    string predict(const vector<float>& input) const;

private:
    struct Node {
        bool isLeaf;
        string label;
        int featureIndex;
        float threshold;
        Node* left;
        Node* right;

        Node() : isLeaf(false), featureIndex(-1), threshold(0.0), left(nullptr), right(nullptr) {}
    };

    Node* root;
    int maxDepth;
    int minSamples;

    Node* buildTree(const vector<vector<float>>& data, const vector<string>& labels, int depth);
    string majorityClass(const vector<string>& labels) const;
    float giniImpurity(const vector<string>& labels) const;
    void splitDataset(const vector<vector<float>>& data, const vector<string>& labels,
                      int feature, float threshold,
                      vector<vector<float>>& leftData, vector<string>& leftLabels,
                      vector<vector<float>>& rightData, vector<string>& rightLabels) const;
    string predictSample(Node* node, const vector<float>& input) const;
    void freeTree(Node* node);
};

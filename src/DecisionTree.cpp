#include "DecisionTree.h"

DecisionTree::DecisionTree(int maxDepth, int minSamples)
    : root(nullptr), maxDepth(maxDepth), minSamples(minSamples) {}

void DecisionTree::train(const vector<vector<float>>& data, const vector<string>& labels) {
    root = buildTree(data, labels, 0);
}

DecisionTree::Node* DecisionTree::buildTree(const vector<vector<float>>& data,
                                            const vector<string>& labels, int depth) {
    Node* node = new Node();

    if (depth >= maxDepth || labels.size() <= minSamples || giniImpurity(labels) == 0.0f) {
        node->isLeaf = true;
        node->label = majorityClass(labels);
        return node;
    }

    int bestFeature = -1;
    float bestThreshold = 0.0f;
    float bestImpurity = numeric_limits<float>::max();
    vector<vector<float>> bestLeft, bestRight;
    vector<string> bestLeftLabels, bestRightLabels;

    for (int feature = 0; feature < data[0].size(); ++feature) {
        for (const auto& row : data) {
            float threshold = row[feature];
            vector<vector<float>> leftData, rightData;
            vector<string> leftLabels, rightLabels;
            splitDataset(data, labels, feature, threshold, leftData, leftLabels, rightData, rightLabels);

            if (leftLabels.empty() || rightLabels.empty()) continue;

            float impurity = (leftLabels.size() * giniImpurity(leftLabels) +
                              rightLabels.size() * giniImpurity(rightLabels)) / labels.size();

            if (impurity < bestImpurity) {
                bestImpurity = impurity;
                bestFeature = feature;
                bestThreshold = threshold;
                bestLeft = leftData;
                bestRight = rightData;
                bestLeftLabels = leftLabels;
                bestRightLabels = rightLabels;
            }
        }
    }

    if (bestFeature == -1) {
        node->isLeaf = true;
        node->label = majorityClass(labels);
        return node;
    }

    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->left = buildTree(bestLeft, bestLeftLabels, depth + 1);
    node->right = buildTree(bestRight, bestRightLabels, depth + 1);

    return node;
}

float DecisionTree::giniImpurity(const vector<string>& labels) const {
    unordered_map<string, int> counts;
    for (const auto& label : labels) {
        counts[label]++;
    }

    float impurity = 1.0f;
    int total = labels.size();
    for (const auto& pair : counts) {
        float prob = (float)pair.second / total;
        impurity -= prob * prob;
    }
    return impurity;
}

string DecisionTree::majorityClass(const vector<string>& labels) const {
    unordered_map<string, int> count;
    for (const string& label : labels)
        count[label]++;
    
    string majority;
    int maxCount = 0;
    for (const auto& pair : count) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            majority = pair.first;
        }
    }
    return majority;
}

void DecisionTree::splitDataset(const vector<vector<float>>& data, const vector<string>& labels,
    int feature, float threshold,
    vector<vector<float>>& leftData, vector<string>& leftLabels,
    vector<vector<float>>& rightData, vector<string>& rightLabels) const {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i][feature] <= threshold) {
                leftData.push_back(data[i]);
                leftLabels.push_back(labels[i]);
            }   
                else{
                rightData.push_back(data[i]);
                rightLabels.push_back(labels[i]);
            }
    }
}

string DecisionTree::predict(const vector<float>& input) const {
    return predictSample(root, input);
}

string DecisionTree::predictSample(Node* node, const vector<float>& input) const {
    if (node->isLeaf)
        return node->label;

    if (input[node->featureIndex] <= node->threshold)
        return predictSample(node->left, input);
    else
        return predictSample(node->right, input);
}

void DecisionTree::freeTree(Node* node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "../src/decisionTree.h"

using namespace std;

void test_decisionTree_training() {
    vector<vector<float>> data = {
        {1.0}, {2.0}, {3.0}, {10.0}, {11.0}
    };
    vector<string> labels = {
        "Low", "Low", "Low", "High", "High"
    };

    decisionTree tree(3, 1);  // maxDepth = 3, minSamples = 1
    tree.train(data, labels);

    string pred1 = tree.predict({1.5});  // Should be Low
    string pred2 = tree.predict({10.5}); // Should be High

    assert(pred1 == "Low");
    assert(pred2 == "High");

    cout << "✅ test_decisionTree_training passed." << endl;
}

int main() {
    test_decisionTree_training();
    return 0;
}
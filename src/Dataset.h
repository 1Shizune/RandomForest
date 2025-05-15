#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
using namespace std;

class dataset {
public:
    vector<vector<float>> data;
    vector<string> label;

    dataset(const vector<vector<float>>& data, const vector<string>& label)
        : data(data), label(label) {}
};

#endif
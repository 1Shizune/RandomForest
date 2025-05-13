#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>

class dataset {
public:
    std::vector<std::vector<float>> data;
    std::vector<std::string> label;

    // Constructor
    dataset(const std::vector<std::vector<float>>& data, const std::vector<std::string>& label)
        : data(data), label(label) {}
};

#endif
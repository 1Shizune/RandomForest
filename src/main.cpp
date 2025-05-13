#include <iostream>
#include <vector>
#include <string>
#include "loadData.h"
#include "decisionTree.h"
#include "RandomForest.h"
#include "Dataset.h"  // Correct inclusion of dataset header

using namespace std;

int main() {
    vector<vector<float>> data;
    vector<string> labels;

    loadDataset("Sleep_health.csv", data, labels); // Load the dataset from file

    dataset ds(data, labels); // Initialize dataset object

    predictionModel model;  // Initialize model and train it
    model.trainModel(data, labels); // Train the model with raw data and labels

    vector<float> testInput = {0, 28, 6.0, 4.0, 30.0, 8.0, 2, 85.0, 3000}; // Sample input

    string prediction = model.predict(testInput); // Make a prediction

    cout << "Prediction: " << prediction << endl; // Output result

    return 0;
}
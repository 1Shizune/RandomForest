#include <iostream>
#include <vector>
#include <string>
#include "loadData.h"
#include "decisionTree.h"
#include "RandomForest.h"
#include "Dataset.h" 
#include "dataProcessor.h"

using namespace std;

int main(int argc, char* argv[]) {
    vector<vector<float>> data;
    vector<string> labels;

    loadDataset("Sleep_health_and_lifestyle_dataset.csv", data, labels);
    cout << "Loaded rows: " << data.size() << ", Labels: " << labels.size() << endl;

    dataProcessor processor(data, labels);
    cout << "Processor initialized." << endl;

    vector<vector<float>> cleanedData;
    for (const auto& row : data) {
        cleanedData.push_back(processor.cleanData(row));  // Use global medians
    }

    vector<vector<float>> normalizedData;
    for (const auto& row : cleanedData) {
        normalizedData.push_back(processor.normalizeData(row));
    }

    predictionModel model;
    cout << "Training model..." << endl;
    model.trainModel(normalizedData, labels);

    // 🔹 Step 5: Define test input
    vector<vector<float>> testInput = {
        {0, 28, -1, 9.0, 60.0, 1.0, 0, 85.0, 8000}, 
        {1, -1, 7.5, 7.0, -1, 3.0, 0, 70.0, 10000},
        {1, 30, -1.0, 8.0, 60.0, 2.0, 0, 70.0, 9000},
        {0, 45, 7.5, 9.0, 50.0, -1.0, 0, 68.0, 10000},
        {1, 25, 8.0, 8.5, 75.0, 3.0, 0, -1.0, 12000}
    };

    vector<string> expected = {
        "Not At Risk",
        "At Risk",
        "Not At Risk",
        "Not At Risk",
        "Not At Risk"
    };

    cout << "Starting prediction on testInput..." << endl;

    for (size_t i = 0; i < testInput.size(); ++i) {
        vector<float> cleaned = processor.cleanData(testInput[i]);  // No label used
        vector<float> normalized = processor.normalizeData(cleaned);

        string pred = model.predict(normalized);
        cout << "Sample " << i + 1 << " Prediction: " << pred
             << " | Expected: " << expected[i] << endl;
    }

    return 0;
}

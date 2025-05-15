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

    loadDataset("src/Sleep_health_and_lifestyle_dataset.csv", data, labels);
    cout << "Loaded rows: " << data.size() << ", Labels: " << labels.size() << endl;

    dataProcessor processor(data, labels);
    cout << "Processor initialized." << endl;

    vector<vector<float>> cleanedData;
    for (const auto& row : data) {
        cleanedData.push_back(processor.cleanData(row));  // Clean missing data in the dataset
    }

    vector<vector<float>> normalizedData; //Make values have a similar scale
    for (const auto& row : cleanedData) {
        normalizedData.push_back(processor.normalizeData(row));
    }

    predictionModel model;
    cout << "Training model..." << endl;
    model.trainModel(normalizedData, labels);

  char repeat;
do {
    vector<float> userInput(9);
    cout << "\nPlease enter your sleep and lifestyle information below:\n";

    cout << "Gender (0 = Male, 1 = Female): ";
    cin >> userInput[0];

    cout << "Age: ";
    cin >> userInput[1];

    cout << "Sleep Duration (hours): ";
    cin >> userInput[2];

    cout << "Quality of Sleep (1-10): ";
    cin >> userInput[3];

    cout << "Physical Activity Level (minutes): ";
    cin >> userInput[4];

    cout << "Stress Level (1-10): ";
    cin >> userInput[5];

    cout << "BMI Category (0 = Normal, 1 = Overweight, 2 = Obese): ";
    cin >> userInput[6];

    cout << "Heart Rate (bpm): ";
    cin >> userInput[7];

    cout << "Daily Steps: ";
    cin >> userInput[8];

    vector<float> cleaned = processor.cleanData(userInput, "global");
    vector<float> normalized = processor.normalizeData(cleaned);

    string prediction = model.predict(normalized);
    cout << "\n Prediction: You are \"" << prediction << "\" for sleep disorder risk.\n";

    cout << "\nWould you like to test another person? (y/n): ";
    cin >> repeat;

} while (repeat == 'y' || repeat == 'Y');

    return 0;
}

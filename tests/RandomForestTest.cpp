#include <iostream>
#include <cassert>
#include "RandomForest.h"

using namespace std;

void testPredictOnEmptyForest() { //Should output nothing, Try to get a prediction when the random forest is empty
    cout << "Running test: Predict on empty forest...\n";

    predictionModel model(5); // Create a model with 5 trees
    vector<float> inputData = {1.0, 2.0, 3.0};

    string prediction = model.predict(inputData);

    cout << "Input: {1.0, 2.0, 3.0}\n";
    cout << "Expected prediction: \"\"\n";
    cout << "Actual prediction:   \"" << prediction << "\"\n";

    assert(prediction == "" && "Expected empty prediction when forest is empty.");
    cout << "Passed\n\n";
}
 
void testTrainWithEmptyData() { //Should output nothing, Try to train the RF when the dataset is empty
    cout << "Running test: Train with empty data...\n";

    predictionModel model(5);
    vector<vector<float>> data;  // Empty dataset
    vector<string> labels;       // Empty labels

    model.trainModel(data, labels);
    vector<float> inputData = {1.0, 2.0};

    string prediction = model.predict(inputData);

    cout << "Training with empty data and labels.\n";
    cout << "Input: {1.0, 2.0}\n";
    cout << "Expected prediction: \"\"\n";
    cout << "Actual prediction:   \"" << prediction << "\"\n";

    assert(prediction == "" && "Expected empty prediction after training with no data.");
    cout << "Passed\n\n";
}

void testSmallDataset() { //Train the model using a dataset that is small
    cout << "Running test: Small dataset training...\n";

    predictionModel model(3);  // 3 trees

    vector<vector<float>> data = {
        {1.0, 2.0},
        {1.1, 2.1},
        {0.9, 1.9}
    };
    vector<string> labels = {
        "healthy",
        "disorder",
        "healthy"
    };

    model.trainModel(data, labels);

    vector<float> testInput = {1.05, 2.05};
    string prediction = model.predict(testInput);

    cout << "Training data: 3 samples\n";
    cout << "Test input: {1.05, 2.05}\n";
    cout << "Expected: \"healthy\" or \"disorder\" (depending on vote)\n";
    cout << "Actual prediction: \"" << prediction << "\"\n";

    assert((prediction == "healthy" || prediction == "disorder") &&
           "Prediction should be one of the known labels.");
    cout << "Passed\n\n";
}

    void testBasicPredictionConsistency() { //test if the tree stops growing given that all labels are equal
    cout << "Running test: Basic prediction consistency...\n";

    predictionModel model(5);  //use 5 trees for redundancy

    
    vector<vector<float>> data = { //all samples are labeled healthy
        {1.0, 2.0},
        {1.1, 2.1},
        {0.9, 1.9},
        {1.2, 2.2},
        {1.05, 2.05}
    };
    vector<string> labels = {
        "healthy",
        "healthy",
        "healthy",
        "healthy",
        "healthy"
    };

    model.trainModel(data, labels);

    
    vector<float> testInput = {10.0, 20.0}; //any input, all trees should predict healthy
    string prediction = model.predict(testInput);

    cout << "Expected: healthy\n";
    cout << "Actual prediction: \"" << prediction << "\"\n";

    assert(prediction == "healthy" && "Prediction should be 'healthy' when all training labels are the same.");
    cout << "Passed\n";
}

void testUnseenInputBehavior() { //Test behavior for when user input is far from training data values
    cout << "Running test: Unseen input behavior...\n";

    predictionModel model(5);
    vector<vector<float>> data = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}
    };
    vector<string> labels = {"healthy", "healthy", "healthy"};

    model.trainModel(data, labels);

    vector<float> newInput = {10.0, 10.0};  // Unseen, far from training data
    string prediction = model.predict(newInput);

    assert(prediction == "healthy" && "Prediction should fallback to dominant label.");
    cout << "Passed\n";
}


int main() {
    testPredictOnEmptyForest();
    testTrainWithEmptyData();
    testSmallDataset();
    testBasicPredictionConsistency();
    testUnseenInputBehavior();
    cout << "All manual tests passed!\n";
    return 0;
}
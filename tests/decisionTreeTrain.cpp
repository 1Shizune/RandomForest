#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "../src/decisionTree.h"
#include "../src/dataProcessor.h"

using namespace std;

string mapToRiskLabel(const string& label) {
    if (label == "None") return "Not At Risk";
    else return "At Risk";
}

void test_risk_prediction() { //test that decisionTree makes the correct prediction given a toy dataset
    cout << " Running test_risk_prediction..." << endl;

    vector<vector<float>> data = {
        {0, 28, 5.9, 4.0, 30.0, 8.0, 2, 85.0, 3000},   // Risk
        {0, 28, 6.2, 6.0, 60.0, 4.0, 0, 75.0, 10000},  // None
        {0, 28, 5.9, 4.0, 30.0, 8.0, 2, 85.0, 3000},   // Risk
        {0, 28, 6.1, 6.0, 42.0, 6.0, 0, 77.0, 4200},   // None
        {1, 30, 5.8, 3.0, 20.0, 9.0, 1, 90.0, 2000},   // Risk
        {1, 26, 6.5, 7.0, 65.0, 3.0, 0, 72.0, 11000},  // None
        {0, 45, 4.5, 2.0, 10.0, 9.0, 2, 95.0, 1500},   // Risk
        {1, 50, 6.0, 5.0, 40.0, 6.0, 0, 80.0, 6000},   // None
        {0, 35, 5.5, 4.0, 25.0, 7.0, 1, 88.0, 3500},   // Risk
        {1, 29, 6.4, 7.0, 55.0, 3.0, 0, 70.0, 9500}    // None
    };

    vector<string> labels = {
        "Sleep Apnea", "None", "Insomnia", "None", "Sleep Apnea",
        "None", "Insomnia", "None", "Sleep Apnea", "None"
    };

    
    for (auto& label : labels) { // Convert detailed labels to risk labels before training
        label = mapToRiskLabel(label);
    }

    decisionTree tree(5, 1);
    tree.train(data, labels);

    struct TestCase {
        vector<float> input;
        string expected;
        string label;
    };

    vector<TestCase> tests = {
        {{0, 28, 6.1, 6.0, 42.0, 6.0, 0, 77.0, 4200}, "Not At Risk", "Typical healthy person"},
        {{0, 28, 6.1, 6.0, 42.0, 6.0, 2, 77.0, 4200}, "Not At Risk", "High BMI, normal everything else"},
        {{1, 30, 5.8, 3.0, 20.0, 9.0, 2, 90.0, 2000}, "At Risk", "Unhealthy female with short sleep"},
        {{0, 45, 4.5, 2.0, 10.0, 9.0, 2, 95.0, 1500}, "At Risk", "Older male with poor sleep quality"},
        {{0, 28, 8.0, 8.0, 10.0, 9.0, 1, 95.0, 6000}, "Not At Risk", "Young guy"},
        {{0, 67, 3.0, 3.0, 5.0, 9.0, 2, 95.0, 2500}, "At Risk", "Old male"},
        {{1, 34, 7.0, 8.0, 30.0, 5.0, 1, 70.0, 6000}, "Not At Risk", "Normal woman"},
        {{1, 31, 4.0, 2.0, 10.0, 9.0, 2, 100.0, 1000}, "At Risk", "Unhealthy woman"},
        {{0, 30, 7.0, 7.0, 30.0, 4.0, 1, 67.0, 6000}, "Not At Risk", "Normal guy"},
        {{0, 28, 5.0, 3.0, 15.0, 9.0, 2, 95.0, 2000}, "At Risk", "Young guy"},
        {{0, 48, 5.0, 3.0, 20.0, 9.0, 1, 95.0, 2500}, "At Risk", "Middle aged"}
    };

    for (const auto& test : tests) {
        string prediction = tree.predict(test.input);

        cout << " Test: " << test.label << endl;
        cout << "   Input: [";
        for (size_t i = 0; i < test.input.size(); ++i)
            cout << test.input[i] << (i < test.input.size() - 1 ? ", " : "");
        cout << "]" << endl;
        cout << "   Prediction: " << prediction << ", Expected: " << test.expected << endl;

        assert(prediction == test.expected);
        cout << "Passed\n" << endl;
    }

    cout << "All risk prediction tests passed successfully!" << endl;
}


void test_invalid_input_handling_with_cleaning() { //Test that decisionTree and dataProcessor correctly deal with nonsensical and missing data
    cout << "\nRunning test_invalid_input_handling_with_cleaning..." << endl;

    vector<vector<float>> data = { //training data
        {0, 28, 7.0, 6.0, 60.0, 3.0, 0, 70.0, 9000},
        {1, 34, 6.8, 7.0, 65.0, 2.0, 0, 68.0, 10000},
        {0, 45, 7.5, 6.0, 50.0, 4.0, 1, 72.0, 8000},
        {1, 30, 6.0, 6.0, 55.0, 3.0, 0, 75.0, 8500},
        {0, 38, 6.5, 7.0, 70.0, 2.0, 0, 65.0, 11000},
        {1, 29, 7.0, 6.0, 58.0, 3.0, 0, 69.0, 9500},
        {0, 50, 6.5, 6.5, 60.0, 2.5, 1, 72.0, 8000},
        {1, 45, 4.5, 3.0, 15.0, 8.0, 2, 95.0, 2000},
        {0, 55, 5.0, 4.0, 20.0, 7.0, 1, 90.0, 2500},
        {1, 62, 4.0, 2.0, 10.0, 9.0, 2, 98.0, 1000},
        {0, 40, 5.5, 4.0, 25.0, 6.0, 2, 88.0, 3000},
        {1, 35, 3.5, 3.0, 18.0, 8.0, 1, 92.0, 1500},
        {0, 48, 5.0, 4.5, 30.0, 7.5, 1, 91.0, 3500},
        {1, 60, 4.2, 3.5, 22.0, 8.5, 2, 93.0, 1800},
        {0, 52, 4.8, 4.0, 28.0, 7.0, 2, 96.0, 2200}
    };

    vector<string> labels = {
        "Not At Risk", "Not At Risk", "Not At Risk", "Not At Risk", "Not At Risk",
        "Not At Risk", "Not At Risk",
        "At Risk", "At Risk", "At Risk", "At Risk", "At Risk", "At Risk", "At Risk", "At Risk"
    };

    
    dataProcessor dp(data, labels); //Initialize dataProcessor with training data (builds medians etc)

    decisionTree tree(3, 1); //Train decision tree as normal
    tree.train(data, labels);

    struct InvalidTestCase {
        vector<float> input;
        string label;
    };

    vector<InvalidTestCase> tests = {
        {{0, 28, 25.0, 6.0, 60.0, 4.0, 0, 70.0, 8000}, "25 hours of sleep, Should be cleaned, Healthy"},
        {{1, 45, 6.0, 5.0, 60.0, 4.0, 0, 0.0, 8000}, "0 heart rate, Should be cleaned, Healthy"},
        {{1, 45, 3.0, 2.0, 15.0, 7.0, 0, 0.0, 1000}, "0 heart rate, Should be cleaned, Unhealthy"},
        {{1, 30, -1.0, 5.0, 50.0, 4.0, 0, 65.0, 7000}, "Negative sleep, Should be cleaned"},
        {{0, 200, 6.0, 6.0, 40.0, 3.0, 0, 75.0, 5000}, "Unrealistic age, Should be cleaned, Healthy"},
        {{1, 60, 4.0, 6.0, 10.0, 7.0, 150.0, 95.0, 1000}, "BMI 150, Should be cleaned, Unhealthy"}
    };

    for (const auto& test : tests) {
        vector<float> cleaned = dp.cleanData(test.input); //Clean nonsensical/missing values using dataProcessor
    
        cout << "Cleaned Input: [";
        for (size_t i = 0; i < cleaned.size(); ++i)
            cout << cleaned[i] << (i < cleaned.size() - 1 ? ", " : "");
        cout << "]" << endl;
    }

        cout << "\nInvalid input handling test with cleaning complete.\n" << endl;
    }

    void test_minimal_training_data() { //test the decision tree (predict) when there are a small amount of examples to look at
    cout << "\nRunning test_minimal_training_data..." << endl;

    vector<vector<float>> data = {
        {0, 30, 6.5, 6.0, 40.0, 3.0, 0, 70.0, 8000},  //Not At Risk
        {1, 45, 4.0, 3.0, 20.0, 8.0, 2, 90.0, 1500}   //At Risk
    };

    vector<string> labels = {"Not At Risk", "At Risk"};

    decisionTree tree(1, 1);
    tree.train(data, labels);

    string prediction1 = tree.predict({0, 30, 6.5, 6.0, 40.0, 3.0, 0, 70.0, 8000});
    string prediction2 = tree.predict({1, 45, 4.0, 3.0, 20.0, 8.0, 2, 90.0, 1500});

    assert(prediction1 == "Not At Risk");
    assert(prediction2 == "At Risk");

    cout << "Passed minimal training test.\n";
}

    void test_input_stability() { //test input with small variations
    cout << "\nRunning test_input_stability..." << endl;

    vector<vector<float>> data = {
        {0, 30, 7.0, 6.0, 50.0, 3.0, 0, 72.0, 8500},
        {1, 29, 4.0, 3.0, 20.0, 8.0, 2, 90.0, 2000}
    };

    vector<string> labels = {"Not At Risk", "At Risk"};

    decisionTree tree(2, 1);
    tree.train(data, labels);

    string pred1 = tree.predict({0, 30, 7.0, 6.0, 50.0, 3.0, 0, 72.0, 8500});
    string pred2 = tree.predict({0, 30, 7.1, 6.0, 50.0, 3.0, 0, 72.0, 8500}); //slight variation

    assert(pred1 == pred2);
    cout << "Stable prediction passed.\n";
}



int main() {
    test_risk_prediction();
    test_invalid_input_handling_with_cleaning();
    test_minimal_training_data();
    test_input_stability();
    
    return 0;
}

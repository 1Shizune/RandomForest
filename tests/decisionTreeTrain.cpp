#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "../src/decisionTree.h"

using namespace std;

void test_deterministic_prediction() {
    cout << " Running test_deterministic_prediction..." << endl;

    vector<vector<float>> data = {
        {0, 28, 5.9, 4.0, 30.0, 8.0, 2, 85.0, 3000},   // Sleep Apnea
        {0, 28, 6.2, 6.0, 60.0, 4.0, 1, 75.0, 10000},  // None
        {0, 28, 5.9, 4.0, 30.0, 8.0, 2, 85.0, 3000},   // Insomnia
        {0, 28, 6.1, 6.0, 42.0, 6.0, 2, 77.0, 4200},   // None
        {1, 30, 5.8, 3.0, 20.0, 9.0, 2, 90.0, 2000},   // Sleep Apnea
        {1, 26, 6.5, 7.0, 65.0, 3.0, 1, 72.0, 11000},  // None
        {0, 45, 4.5, 2.0, 10.0, 9.0, 2, 95.0, 1500},   // Insomnia
        {1, 50, 6.0, 5.0, 40.0, 6.0, 1, 80.0, 6000},   // None
        {0, 35, 5.5, 4.0, 25.0, 7.0, 2, 88.0, 3500},   // Sleep Apnea
        {1, 29, 6.4, 7.0, 55.0, 3.0, 1, 70.0, 9500}    // None
    };

    vector<string> labels = {
        "Sleep Apnea", "None", "Insomnia", "None", "Sleep Apnea",
        "None", "Insomnia", "None", "Sleep Apnea", "None"
    };

    decisionTree tree(5, 1);
    tree.train(data, labels);

    struct TestCase {
        vector<float> input;
        string expected;
        string label;
    };

    vector<TestCase> tests = { //Gender,Age,SleepDuration,QualityOfSleep,PhysicalActivity,StressLevel,BMICategory,HeartRate,DailySteps
        {{0, 28, 6.1, 6.0, 42.0, 6.0, 2, 77.0, 4200}, "None", "Typical healthy person"},
        {{1, 30, 5.8, 3.0, 20.0, 9.0, 2, 90.0, 2000}, "Sleep Apnea", "Unhealthy female with short sleep"},
        {{0, 45, 4.5, 2.0, 10.0, 9.0, 2, 95.0, 1500}, "Insomnia", "Older male with poor sleep quality"},
        {{0, 28, 8.0, 8.0, 10.0, 9.0, 1, 95.0, 6000}, "None", "Young guy"},
        {{0, 67, 3.0, 3.0, 5.0, 9.0, 2, 95.0, 2500}, "Insomnia", "Old male"},
        {{1, 34, 7.0, 8.0, 30.0, 5.0, 1, 70.0, 6000}, "None", "Normal woman"},
        {{1, 31, 4.0, 2.0, 10.0, 9.0, 2, 100.0, 1000}, "Sleep Apnea", "Unhealthy woman"},
        {{0, 30, 7.0, 7.0, 30.0, 4.0, 1, 67.0, 6000}, "None", "Norman"},
        {{0, 28, 6.0, 3.0, 15.0, 9.0, 2, 95.0, 2000}, "None", "Young guy"},
        {{0, 48, 5.0, 3.0, 20.0, 9.0, 1, 95.0, -1}, "Insomnia", "Middle aged"}
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

    cout << "All deterministic prediction tests passed successfully!" << endl;
}
int main() {
    test_deterministic_prediction();
    return 0;
}

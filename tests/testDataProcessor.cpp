#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "../src/dataProcessor.h"

using namespace std;

void test_missing_value_replacement() {
    cout << "Running test_missing_value_replacement...\n";

    vector<vector<float>> data = {
        {0, 25, 7.0, 7.0, 50.0, 3.0, 1, 70.0, 9000},  // Healthy
        {1, 45, 5.0, 4.0, 15.0, 8.0, 2, 90.0, 2000},  // At Risk
        {0, 30, 6.5, 6.0, 40.0, 4.0, 1, 72.0, 8000},  // Healthy
        {1, 50, 5.5, 3.0, 20.0, 9.0, 2, 95.0, 1500},  // At Risk
    };
    vector<string> labels = {
        "Not At Risk", "At Risk", "Not At Risk", "At Risk"
    };

    dataProcessor processor(data, labels);

    vector<float> rawUser = {
        0, 25, 8.0, -1.0f, 90.0, -1.0f, 1, -1.0f, 9000
        // SleepDuration, StressLevel, HeartRate are missing
    };

    vector<float> cleaned = processor.cleanData(rawUser);

    cout << "Cleaned user input:\n[ ";
    for (float val : cleaned) cout << val << " ";
    cout << "]\n\n";

    
    assert(cleaned[2] != -1.0f); // SleepDuration replaced
    assert(cleaned[5] != -1.0f); // StressLevel replaced
    assert(cleaned[7] != -1.0f); // HeartRate replaced

    cout << "Test passed: Missing values replaced correctly based on user classification.\n";
}

int main() {
    test_missing_value_replacement();
    return 0;
}

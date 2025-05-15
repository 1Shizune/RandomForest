#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include "loadData.h"

using namespace std;

unordered_map<string, int> genderMap = {{"Male", 0}, {"Female", 1}};
unordered_map<string, int> bmiMap = {{"Normal", 0}, {"Overweight", 1}, {"Obese", 2}};
unordered_map<string, int> disorderMap = {{"None", 0}, {"Sleep Apnea", 1}, {"Insomnia", 2}};


float safeParseFloat(const string& token) { //Helper to safely parse float, returns -1.0f if empty or invalid
    if (token.empty()) return -1.0f;
    try {
        return stof(token);
    } catch (...) {
        return -1.0f;
    }
}


float safeParseCategory(const string& token, const unordered_map<string, int>& map) { //Helper to safely parse categorical mapping, returns -1.0f if not found
    auto it = map.find(token);
    if (it != map.end()) return static_cast<float>(it->second);
    else return -1.0f;
}

void loadDataset(const string& filename,
    vector<vector<float>>& features,
    vector<string>& labels) {
    
    ifstream file(filename);
    string line;

    getline(file, line); //skip header

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<float> row;

        
        getline(ss, token, ','); //Person ID (skip)

       
        getline(ss, token, ',');  //Gender
        row.push_back(safeParseCategory(token, genderMap));

        
        getline(ss, token, ','); //Age
        row.push_back(safeParseFloat(token));

        
        getline(ss, token, ','); //Occupation (skip)

        
        getline(ss, token, ','); //Sleep Duration
        row.push_back(safeParseFloat(token));

        
        getline(ss, token, ','); //Quality of Sleep
        row.push_back(safeParseFloat(token));

        
        getline(ss, token, ','); //Physical Activity Level
        row.push_back(safeParseFloat(token));

       
        getline(ss, token, ',');  //Stress Level
        row.push_back(safeParseFloat(token));

        
        getline(ss, token, ','); //BMI Category
        row.push_back(safeParseCategory(token, bmiMap));

        
        getline(ss, token, ','); //Blood Pressure (skip)

       
        getline(ss, token, ',');  //Heart Rate
        row.push_back(safeParseFloat(token));

       
        getline(ss, token, ',');  //Daily Steps
        row.push_back(safeParseFloat(token));

        
        getline(ss, token, ','); //Sleep Disorder label
        if (token == "None") {
            labels.push_back("Not At Risk");
        } else {
            labels.push_back("At Risk");
        }

        features.push_back(row);
    }

    file.close();
}

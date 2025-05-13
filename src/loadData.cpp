#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include "loadData.h"

using namespace std;

unordered_map<string, int> genderMap = {{"Male", 0}, {"Female", 1}};
unordered_map<string, int> bmiMap = {{"Normal", 0}, {"Overweight", 1}, {"Obese", 2}};
unordered_map<string, int> disorderMap = {{"None", 0}, {"Sleep Apnea", 1}, {"Insomnia", 2}};

void loadDataset(const string& filename,
                 vector<vector<float>>& features,
                 vector<string>& labels) {
    ifstream file(filename);
    string line;

    getline(file, line); // skip header

    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<float> row;

        getline(ss, token, ','); // Person ID — skip

        
        getline(ss, token, ','); // Gender (convert to float)
        row.push_back(genderMap[token]);

       
        getline(ss, token, ',');  // Age
        row.push_back(stof(token));

        
        getline(ss, token, ','); // Occupation

       
        getline(ss, token, ',');  // Sleep Duration
        row.push_back(stof(token));

       
        getline(ss, token, ',');  // Quality of Sleep
        row.push_back(stof(token));

        
        getline(ss, token, ','); // Physical Activity Level
        row.push_back(stof(token));

     
        getline(ss, token, ',');    // Stress Level
        row.push_back(stof(token));

       
        getline(ss, token, ',');  // BMI Category
        row.push_back(bmiMap[token]);

        
        getline(ss, token, ','); // Blood Pressure — skip or split (optional)

        
        getline(ss, token, ','); // Heart Rate
        row.push_back(stof(token));

        
        getline(ss, token, ','); // Daily Steps
        row.push_back(stof(token));

        
        getline(ss, token, ','); // Sleep Disorder (label)
        labels.push_back(token);

        features.push_back(row);
    }

    file.close();
}

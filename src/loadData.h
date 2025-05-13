#ifndef LOADDATA_H
#define LOADDATA_H

#include <vector>
#include <string>

using namespace std;

// Declare the loadDataset function
void loadDataset(const string& filename, vector<vector<float>>& features, vector<string>& labels);

#endif // LOADDATA_H
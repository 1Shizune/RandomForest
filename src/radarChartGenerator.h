#ifndef RADARCHARTGENERATOR_H
#define RADARCHARTGENERATOR_H

#include <vector>
using namespace std;

class radarChartGenerator {
public:
    radarChartGenerator();
    void generatorChart(vector<float>& userData, vector<float>& healthyData);
};

#endif
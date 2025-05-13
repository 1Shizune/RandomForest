#ifndef RADARCHARTGENERATOR_H
#define RADARCHARTGENERATOR_H

#include <vector>

class radarChartGenerator {
public:
    radarChartGenerator();
    void generatorChart(std::vector<float>& userData, std::vector<float>& healthyData);
};

#endif
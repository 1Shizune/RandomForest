#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "../src/dataProcessor.h"
#include "../src/RandomForest.h"

std::string mapLabel(const std::string& label) {
    return (label == "None") ? "Not At Risk" : "At Risk";
}

void loadDataset(const std::string& path, std::vector<std::vector<float>>& data, std::vector<std::string>& labels) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }

    std::string line;
    getline(file, line);

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 13) continue;

        std::vector<float> row;
        row.push_back(tokens[1] == "Male" ? 0.0f : 1.0f);
        row.push_back(std::stof(tokens[2]));
        row.push_back(std::stof(tokens[4]));
        row.push_back(std::stof(tokens[5]));
        row.push_back(std::stof(tokens[6]));
        row.push_back(std::stof(tokens[7]));

        if (tokens[8] == "Normal") row.push_back(0.0f);
        else if (tokens[8] == "Overweight") row.push_back(1.0f);
        else row.push_back(2.0f);

        row.push_back(std::stof(tokens[10]));
        row.push_back(std::stof(tokens[11]));

        data.push_back(row);
        labels.push_back(mapLabel(tokens[12]));
    }

    std::cout << "Loaded rows: " << data.size() << ", Labels: " << labels.size() << std::endl;
}

int main() {
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Sleep Risk GUI", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window.\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    ImGui::StyleColorsDark();

    std::vector<std::vector<float>> data;
    std::vector<std::string> labels;
    loadDataset("../src/Sleep_health_and_lifestyle_dataset.csv", data, labels);
    dataProcessor processor(data, labels);
    predictionModel model;
    model.trainModel(data, labels);
    std::cout << "Model trained." << std::endl;

    float gender = 0, age = 25, sleepDuration = 7, sleepQuality = 5, activity = 30, stress = 4, bmi = 0, heartRate = 75, steps = 8000;
    std::string prediction;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowSize(ImVec2(600, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(100, 50), ImGuiCond_FirstUseEver);

        ImGui::Begin("Sleep Disorder Predictor", nullptr, ImGuiWindowFlags_NoCollapse);

        const char* genderItems[] = { "Male", "Female" };
        const char* bmiItems[] = { "Normal", "Overweight", "Obese" };
        int genderIndex = static_cast<int>(gender);
        int bmiIndex = static_cast<int>(bmi);
        int ageInt = static_cast<int>(age);
        int sleepQualityInt = static_cast<int>(sleepQuality);
        int stressInt = static_cast<int>(stress);

        ImGui::Text("Fill in the information below:");
        ImGui::Spacing();
        ImGui::Separator();

        ImGui::Combo("Gender", &genderIndex, genderItems, IM_ARRAYSIZE(genderItems));
        gender = static_cast<float>(genderIndex);

        ImGui::InputInt("Age", &ageInt); age = static_cast<float>(ageInt);
        ImGui::InputFloat("Sleep Duration", &sleepDuration);
        ImGui::InputInt("Sleep Quality (1-10)", &sleepQualityInt); sleepQuality = static_cast<float>(sleepQualityInt);
        ImGui::InputFloat("Physical Activity", &activity);
        ImGui::InputInt("Stress Level (1-10)", &stressInt); stress = static_cast<float>(stressInt);

        ImGui::Combo("BMI Category", &bmiIndex, bmiItems, IM_ARRAYSIZE(bmiItems));
        bmi = static_cast<float>(bmiIndex);

        ImGui::InputFloat("Heart Rate", &heartRate);
        ImGui::InputFloat("Steps", &steps);

        if (ImGui::Button("Predict", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            std::vector<float> input = { gender, age, sleepDuration, sleepQuality, activity, stress, bmi, heartRate, steps };
            std::vector<float> cleaned = processor.cleanData(input);
            std::vector<float> normalized = processor.normalizeData(cleaned);
            prediction = model.predict(normalized);
            std::cout << "Prediction result: [" << prediction << "]" << std::endl;
        }

        if (!prediction.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("Prediction: You are \"%s\" for sleep disorder risk.", prediction.c_str());

            if (prediction == "At Risk") {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextWrapped("Recommendations:");
    ImGui::BulletText("Establish a consistent sleep schedule. Go to sleep and wake up around the same time each day, even on the weekends.");
    ImGui::BulletText("Avoid caffeine, nicotine, and alcohol close to your bedtime. Although alcohol can make it easier to fall asleep, it can cause you to have a sleep that tends to be lighter than normal. This makes it more likely that you will wake up during the night.");
    ImGui::BulletText("Try relaxation techniques such as deep breathing or meditation");
    ImGui::BulletText("Make your bedroom sleep friendly. Sleep in a cool, quiet, dark place. Avoid watching TV or looking at electronic devices, as the light from these sources can disrupt your sleep-wake cycle.");
    ImGui::BulletText("Get regular physical activity during the daytime (at least 5 to 6 hours before going to bed). Exercising close to bedtime can make it harder to fall asleep.");

    ImGui::Spacing();
    ImGui::TextWrapped("Helpful Resources:");
    if (ImGui::Selectable("National Sleep Foundation - sleepfoundation.org")) {
        ImGui::SetClipboardText("https://www.sleepfoundation.org");
    }
    if (ImGui::Selectable("CDC Sleep and Sleep Disorders - cdc.gov/sleep")) {
        ImGui::SetClipboardText("https://www.cdc.gov/sleep");
    }
    ImGui::TextWrapped("(Link copied to clipboard when clicked)");
}
        }

        ImGui::End();

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

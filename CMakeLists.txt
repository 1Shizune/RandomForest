cmake_minimum_required(VERSION 3.14)
project(SleepDisorderPredictorGUI)

set(CMAKE_CXX_STANDARD 17)

# Set paths to ImGui and GLAD (adjust if different)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/gui/imgui)
set(GLAD_DIR ${CMAKE_SOURCE_DIR}/gui/glad)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# GLFW (assumes installed via Homebrew or system-wide)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

include_directories(
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GLAD_DIR}/include
    ${SRC_DIR}
)

add_executable(guiApp
    gui/main_gui.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    ${GLAD_DIR}/src/glad.c
    ${SRC_DIR}/dataProcessor.cpp
    ${SRC_DIR}/RandomForest.cpp
    ${SRC_DIR}/decisionTree.cpp
)

target_link_libraries(guiApp
    ${GLFW_LIBRARIES}
    ${CMAKE_DL_LIBS}
    ${GLFW_STATIC_LIBRARIES}
)

# Required for macOS frameworks
if(APPLE)
    target_link_libraries(guiApp "-framework OpenGL")
endif()

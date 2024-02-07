cmake_minimum_required(VERSION 3.19)
include(FetchContent)

project(SmockingSolver)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SRC_FILES *.cpp Optiz/src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Optiz/include)

execute_process(COMMAND python3.10 -m pybind11 --include OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYBIND_INCLUDE)
# In case the above doesn't work, set the include file from the python in the Blender release.
# set(PYBIND_INCLUDE "-I/Applications/Blender.app/Contents/Resources/3.4/python/include/python3.10 -I/Applications/Blender.app/Contents/Resources/3.4/python/lib/python3.10/site-packages/pybind11/include")

# Proper way to generate the link flags, doesn't work for the python release in Blender.
# execute_process(COMMAND python3-config --ldflags OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYTHON_LD_FLAGS)

set(PYTHON_LD_FLAGS "-ldl   -framework CoreFoundation -L/Applications/Blender.app/Contents/Resources/4.0/python/lib/python3.10/config-3.10-darwin")

# Get compilation flags.
execute_process(COMMAND python3.10-config --cflags OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYTHON_CFLAGS)
# And the output suffix.
execute_process(COMMAND python3.10-config --extension-suffix OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE OUTPUT_SUFFIX)

#-flto=thin
SET ( CMAKE_CXX_FLAGS "-Xlinker -undefined -Xlinker dynamic_lookup  -fPIC ${CMAKE_CXX_FLAGS} ${PYBIND_INCLUDE} ${PYTHON_LD_FLAGS} ${PYTHON_CFLAGS}")

# The proper way to handle warnings (ignore them).
add_definitions(-w)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "cpp_smocking_solver")
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ${OUTPUT_SUFFIX})
 
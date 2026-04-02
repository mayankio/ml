cmake_minimum_required(VERSION 3.16)
project(edu_ml_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(edu_ml ${LIB_SOURCES})
target_include_directories(edu_ml PUBLIC include)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE edu_ml)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

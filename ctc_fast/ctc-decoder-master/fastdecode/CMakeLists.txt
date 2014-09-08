cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(fastdecode)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_PREFIX_PATH /afs/cs.stanford.edu/u/zxie/libs ${CMAKE_PREFIX_PATH})

include_directories("/afs/cs.stanford.edu/u/zxie/libs/include")
link_directories("/afs/cs.stanford.edu/u/zxie/libs/lib")

find_package(Boost 1.53.0 EXACT COMPONENTS python timer)
if (Boost_FOUND)
    add_definitions( "-DHAS_BOOST" )
    message("--Boost Configuration--")
    message(${Boost_LIBRARIES})
    message(${Boost_INCLUDE_DIR})
endif()

find_package(PythonLibs REQUIRED)

# OpenMP
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/modules ${CMAKE_MODULE_PATH})
find_package(NumPy REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

add_library(fastdecode SHARED decoder.cpp lm.cpp prefix_tree.cpp)
target_link_libraries(fastdecode ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(fastdecode PROPERTIES PREFIX "")

add_executable(test_decode test_decode.cpp decoder.cpp lm.cpp prefix_tree.cpp)
target_link_libraries(test_decode ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(test_lm test_lm.cpp lm.cpp)
target_link_libraries(test_lm ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(test_prefix_tree test_prefix_tree.cpp prefix_tree.cpp lm.cpp)
target_link_libraries(test_prefix_tree ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

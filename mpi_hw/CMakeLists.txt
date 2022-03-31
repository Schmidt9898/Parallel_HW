cmake_minimum_required(VERSION 3.10)



# set the project name
project(parallel)

set(CMAKE_CXX_STANDARD 17)
#set ( CMAKE_C_FLAGS "fexceptions")
#set ( CMAKE_CXX_FLAGS "-fopenmp -O3 -fopt-info-vec")
set ( CMAKE_CXX_FLAGS "-O3")
# add the executable
add_executable(run main.cpp)

set(include
${CMAKE_SOURCE_DIR} 
)

target_include_directories(run PUBLIC ${include} )
#target_link_libraries(holozer PUBLIC )

#target_link_directories()


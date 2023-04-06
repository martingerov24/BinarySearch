cmake_minimum_required(VERSION 3.20)
project(fmi_binary_search LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-mavx)
add_compile_options(-mavx2)
add_compile_options(-msse4.1)
add_compile_options(-mavx512vl)


set(SOURCES 
    main.cpp 
)

add_executable(${PROJECT_NAME} ${SOURCES})

# did not want to change the file name and location of the input files, so instead of copying the 
# whole directory, I copy file by file the all the input files  
file(GLOB ASSET_FILES "${CMAKE_SOURCE_DIR}/assets/*.bsearch")
foreach(file_path ${ASSET_FILES})
    get_filename_component(FILE_NAME ${file_path} NAME)
    message("Found file: ${file_path} ${FILE_NAME}")
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${file_path}
            ${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}
    )
endforeach()


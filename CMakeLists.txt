
cmake_minimum_required(VERSION 3.25)

project(RocketX C)

add_executable(RocketX
    src/main.c
)

set_target_properties(RocketX PROPERTIES
    C_STANDARD 17
    C_EXTENSIONS ON
    OUTPUT_NAME "Rocket-X"
)

target_compile_options(RocketX PRIVATE
    -Wall
    -Wextra
    -Werror
)

xsdk_add_shaders(RocketX
    src/rocket.vert
    src/rocket.frag
)

xsdk_add_xex(RocketX)

set(SOURCE_FILES
    src/dataset.cpp
    src/layer.cpp
    src/multilayer_perceptron.cpp
    src/perceptron.cpp
    src/trainer.cpp
)

set(PUBLIC_HEADER_FILES
    include/neural/activation_functions.h
    include/neural/assert.h
    include/neural/dataset.h
    include/neural/defines.h
    include/neural/layer.h
    include/neural/multilayer_perceptron.h
    include/neural/perceptron.h
    include/neural/trainer.h
)

add_library(NeuralLib STATIC
    ${SOURCE_FILES}
    ${PRIVATE_HEADER_FILES}
    ${PUBLIC_HEADER_FILES}
)

target_include_directories(NeuralLib
    PUBLIC
        include
)

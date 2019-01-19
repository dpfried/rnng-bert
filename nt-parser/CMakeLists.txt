PROJECT(cnn:nt-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

foreach(TARGET
        nt-parser
        nt-parser-gen
        tensorflow-test
        word-featurizer-test
        )

    ADD_EXECUTABLE(${TARGET} ${TARGET}.cc eval.cc oracle.cc pretrained.cc stack.h tree.h nt-parser.h streaming-statistics.h training-position.h utils.h)
    target_link_libraries(${TARGET} cnn ${Boost_LIBRARIES} z)

    if(UNIX AND NOT APPLE)
        target_link_libraries(${TARGET} rt)
    endif()

    if (WITH_CUDA_BACKEND)
        add_dependencies(${TARGET} cnncuda)
        target_link_libraries(${TARGET} cnncuda)
        CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
    endif(WITH_CUDA_BACKEND)

endforeach()


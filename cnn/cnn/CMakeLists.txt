# ########## cnn library ##########
# Sources:
set(cnn_library_SRCS
    cfsm-builder.cc
    cnn.cc
    conv.cc
    deep-lstm.cc
    devices.cc
    dict.cc
    dim.cc
    exec.cc
    expr.cc
    fast-lstm.cc
    grad-check.cc
    graph.cc
    gru.cc
    hsm-builder.cc
    init.cc
    lstm.cc
    mem.cc
    model.cc
    mp.cc
    nodes.cc
    nodes-common.cc
    param-nodes.cc
    rnn.cc
    rnn-state-machine.cc
    saxe-init.cc
    shadow-params.cc
    tensor.cc
    training.cc
)

# Headers:
set(cnn_library_HDRS
    aligned-mem-pool.h
    cfsm-builder.h
    c2w.h
    cnn.h
    conv.h
    cuda.h
    devices.h
    dict.h
    dim.h
    exec.h
    expr.h
    fast-lstm.h
    functors.h
    gpu-kernels.h
    gpu-ops.h
    graph.h
    gru.h
    hsm-builder.h
    init.h
    lstm.h
    mem.h
    model.h
    mp.h
    nodes.h
    param-nodes.h
    random.h
    rnn-state-machine.h
    rnn.h
    saxe-init.h
    shadow-params.h
    simd-functors.h
    tensor.h
    timing.h
    training.h
)

set(cnn_gpu_SRCS
    cuda.cc
    gpu-ops.cu)

# if(WITH_CUDA_BACKEND)
#   list(APPEND cnn_library_SRCS
#        cuda.cc)
# endif(WITH_CUDA_BACKEND)

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} tests/*.cc)

#foreach(test_src ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
#  get_filename_component(testName ${test_src} NAME_WE)

  #Add compile target
#  add_executable(${testName} ${test_src})

  #link to Boost libraries AND your targets and dependencies
#  target_link_libraries(${testName} cnn ${LIBS})

#  set_target_properties(${testName} PROPERTIES
#      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin)

  #Finally add it to test execution -
  #Notice the WORKING_DIRECTORY and COMMAND
#  add_test(NAME ${testName}
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests.bin/${testName} )
#endforeach(test_src)

if(WITH_CUDA_BACKEND)
  set(CUDA_SEPARABLE_COMPILATION ON)
  list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-std=c++11;-O2;-DVERBOSE;-Xcompiler;-fpic;-DEIGEN_USE_GPU;-DHAVE_CUDA;-D_FORCE_INLINES")
  if(CUDNN_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "-DHAVE_CUDNN")
  endif()
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  # cuda_add_library(cnncuda STATIC gpu-ops.cu)
  # cuda_add_library(cnncuda_shared SHARED gpu-ops.cu)
  #cuda_add_library(cnn gpu-ops.cu)
  #cuda_add_library(cnn gpu-ops.cu)
  cuda_add_library(cnn ${cnn_library_SRCS} ${cnn_library_HDRS} ${cnn_gpu_SRCS} OPTIONS --compiler-options "-fPIC")
  #add_definitions(-DHAVE_CUDA)
  set_target_properties(cnn PROPERTIES
                        COMPILE_DEFINITIONS HAVE_CUDA)
  if(CUDNN_FOUND)
    target_compile_definitions(dynet PRIVATE HAVE_CUDNN)
  endif()
  cuda_add_cublas_to_target(cnn)
  target_link_libraries(cnn ${LIBS})
else()
  add_library(cnn ${cnn_library_SRCS} ${cnn_library_HDRS})
  target_link_libraries(cnn ${LIBS})
endif(WITH_CUDA_BACKEND)

install(FILES ${cnn_library_HDRS} DESTINATION include/cnn)
install(TARGETS cnn DESTINATION lib)
if(WITH_CUDA_BACKEND)
  install(TARGETS cnn DESTINATION lib)
endif(WITH_CUDA_BACKEND)

# target_compile_features(cnn PRIVATE cxx_range_for)


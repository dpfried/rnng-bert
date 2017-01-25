#ifndef CNN_EIGEN_INIT_H
#define CNN_EIGEN_INIT_H

namespace cnn {

void Initialize(int& argc, char**& argv, unsigned random_seed = 0, bool shared_parameters = false);
void Cleanup();

  unsigned global_random_seed = 0;

} // namespace cnn

#endif

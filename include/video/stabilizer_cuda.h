#ifndef STABILIZER_CUDA_H
#define STABILIZER_CUDA_H

#include <vector>

namespace vs {
    std::vector<float> boxFilterConvolveCUDA(const std::vector<float>& path, int r);
}

#endif // STABILIZER_CUDA_H

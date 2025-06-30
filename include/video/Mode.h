#ifndef MODE_H
#define MODE_H

#include <opencv2/opencv.hpp>

namespace vs {
    class Mode {
    public:
        struct Parameters {
            int width;
            int height;
            bool optimizeFps;
            bool useCuda;
            bool enhancerEnabled;
            bool rollCorrectionEnabled;
            bool stabilizationEnabled;
            bool trackerEnabled;
        };
    };
}

#endif // MODE_H

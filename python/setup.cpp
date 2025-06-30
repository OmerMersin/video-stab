#include <pybind11/pybind11.h>
#include "video/stabilizer.h"

namespace py = pybind11;

PYBIND11_MODULE(video_stab, m) {
    py::class_<video::Stabilizer>(m, "Stabilizer")
        .def(py::init<>())
        .def("stabilize", &video::Stabilizer::stabilize);
}

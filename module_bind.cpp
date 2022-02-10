#include <pybind11/pybind11.h>
// conversions between std::map<> and the Python dict data structures are automatically enabled.
#include <pybind11/stl.h>
namespace py = pybind11;

#include "cpp_tokenizer.cc"

PYBIND11_MODULE(cpp_tokenizer, m)
{
    m.doc() = "pybind11 tokenizer plugin";

    m.def("deal_batch", &deal_batch, "do tokenize task",
          py::arg("feature_arr"), py::arg("vocab"), py::arg("length_limit"));
}

/*
c++ -O3 -Wall -shared -std=c++11 -fPIC -I./include/ $(python3 -m pybind11 --includes) module_bind.cpp -o cpp_tokenizer$(python3-config --extension-suffix)

python test.py
*/

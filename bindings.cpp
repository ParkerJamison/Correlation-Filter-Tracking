#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CFT_Track.hpp"
#include "TrackID.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
cv::Mat numpy_to_mat(py::handle input) {
    auto array =
        py::array_t<unsigned char, py::array::c_style | py::array::forcecast>::ensure(input);
    if (!array) {
        throw std::runtime_error("Frame must be a contiguous uint8 NumPy array");
    }

    py::buffer_info buf = array.request();
    if (buf.ndim != 2 && buf.ndim != 3) {
        throw std::runtime_error("Frame must be 2D grayscale or 3D color");
    }

    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    int channels = (buf.ndim == 3) ? static_cast<int>(buf.shape[2]) : 1;

    int type;
    switch (channels) {
        case 1:
            type = CV_8UC1;
            break;
        case 3:
            type = CV_8UC3;
            break;
        case 4:
            type = CV_8UC4;
            break;
        default:
            throw std::runtime_error("Frame must have 1, 3, or 4 channels");
    }

    return cv::Mat(rows, cols, type, buf.ptr).clone();
}

cv::Rect object_to_rect(py::handle value) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(value);
    if (seq.size() != 4) {
        throw std::runtime_error("Bounding box must be a 4-item sequence: (x, y, w, h)");
    }

    return cv::Rect(
        seq[0].cast<int>(),
        seq[1].cast<int>(),
        seq[2].cast<int>(),
        seq[3].cast<int>());
}

py::tuple rect_to_tuple(const cv::Rect& rect) {
    return py::make_tuple(rect.x, rect.y, rect.width, rect.height);
}

template <typename T>
py::array_t<T> mat_to_numpy_typed(const cv::Mat& mat) {
    cv::Mat contiguous = mat.isContinuous() ? mat : mat.clone();
    int channels = contiguous.channels();

    std::vector<py::ssize_t> shape;
    if (channels == 1) {
        shape = {contiguous.rows, contiguous.cols};
    } else {
        shape = {contiguous.rows, contiguous.cols, channels};
    }

    py::array_t<T> array(shape);
    std::memcpy(array.mutable_data(), contiguous.data,
                static_cast<size_t>(contiguous.total() * contiguous.elemSize()));
    return array;
}

py::array mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array_t<unsigned char>();
    }

    switch (mat.depth()) {
        case CV_8U:
            return mat_to_numpy_typed<unsigned char>(mat);
        case CV_32F:
            return mat_to_numpy_typed<float>(mat);
        case CV_64F:
            return mat_to_numpy_typed<double>(mat);
        default:
            throw std::runtime_error("Unsupported cv::Mat depth for NumPy conversion");
    }
}

py::list mat_vector_to_list(const std::vector<cv::Mat>& mats) {
    py::list result;
    for (const cv::Mat& mat : mats) {
        result.append(mat_to_numpy(mat));
    }
    return result;
}

Track init_tracking(CFT& self, py::handle frame, py::object bbox) {
    cv::Mat mat = numpy_to_mat(frame);
    if (bbox.is_none()) {
        return self.initTracking(mat);
    }

    return self.initTracking(mat, object_to_rect(bbox));
}

int update_tracking(CFT& self, py::handle frame, Track& track) {
    cv::Mat mat = numpy_to_mat(frame);
    return self.updateTracking(mat, track);
}

void init_bbox(Track& self, py::handle frame, py::handle bbox) {
    self.initBBox(numpy_to_mat(frame), object_to_rect(bbox));
}
}  // namespace

PYBIND11_MODULE(cft_tracker, m) {
    m.doc() = "CFT and Track classes exposed via pybind11";

    py::class_<CFT>(m, "CFT")
        .def(py::init<>())
        .def(py::init<double, int, int>(),
             py::arg("learning_rate"), py::arg("sigma"), py::arg("num_train"))
        .def("initTracking", &init_tracking,
             py::arg("frame"), py::arg("bbox") = py::none(),
             "Initialize a Track from a frame. Pass bbox=(x, y, w, h) to skip ROI UI.")
        .def("init_tracking", &init_tracking,
             py::arg("frame"), py::arg("bbox") = py::none(),
             "Initialize a Track from a frame. Pass bbox=(x, y, w, h) to skip ROI UI.")
        .def("updateTracking", &update_tracking,
             py::arg("frame"), py::arg("track"))
        .def("update_tracking", &update_tracking,
             py::arg("frame"), py::arg("track"));

    py::class_<Track, CFT>(m, "Track")
        .def(py::init<>())
        .def("initBBox", &init_bbox, py::arg("frame"), py::arg("bbox"))
        .def("init_bbox", &init_bbox, py::arg("frame"), py::arg("bbox"))
        .def("updateBBox",
             [](Track& self, int dx, int dy, py::handle bounds) {
                 self.updateBBox(dx, dy, object_to_rect(bounds));
             },
             py::arg("dx"), py::arg("dy"), py::arg("bounds"))
        .def("update_bbox",
             [](Track& self, int dx, int dy, py::handle bounds) {
                 self.updateBBox(dx, dy, object_to_rect(bounds));
             },
             py::arg("dx"), py::arg("dy"), py::arg("bounds"))
        .def("getBBox", [](const Track& self) { return rect_to_tuple(self.getBBox()); })
        .def("get_bbox", [](const Track& self) { return rect_to_tuple(self.getBBox()); })
        .def("getDisplayBBox",
             [](const Track& self) { return rect_to_tuple(self.getDisplayBBox()); })
        .def("get_display_bbox",
             [](const Track& self) { return rect_to_tuple(self.getDisplayBBox()); })
        .def("getSearchArea",
             [](const Track& self) { return rect_to_tuple(self.getSearchArea()); })
        .def("get_search_area",
             [](const Track& self) { return rect_to_tuple(self.getSearchArea()); })
        .def("getImageBounds",
             [](const Track& self) { return rect_to_tuple(self.getImageBounds()); })
        .def("get_image_bounds",
             [](const Track& self) { return rect_to_tuple(self.getImageBounds()); })
        .def("cropForSearch",
             [](Track& self, py::handle frame) {
                 return mat_to_numpy(self.cropForSearch(numpy_to_mat(frame)));
             },
             py::arg("frame"))
        .def("crop_for_search",
             [](Track& self, py::handle frame) {
                 return mat_to_numpy(self.cropForSearch(numpy_to_mat(frame)));
             },
             py::arg("frame"))
        .def("cropForROI",
             [](Track& self, py::handle frame) {
                 return mat_to_numpy(self.cropForROI(numpy_to_mat(frame)));
             },
             py::arg("frame"))
        .def("crop_for_roi",
             [](Track& self, py::handle frame) {
                 return mat_to_numpy(self.cropForROI(numpy_to_mat(frame)));
             },
             py::arg("frame"))
        .def("updateFilter", &Track::updateFilter,
             py::arg("learning_rate"), py::arg("long_term"))
        .def("update_filter", &Track::updateFilter,
             py::arg("learning_rate"), py::arg("long_term"))
        .def_readwrite("psrFlag", &Track::psrFlag)
        .def_readwrite("psr_flag", &Track::psrFlag)
        .def_property_readonly("A", [](const Track& self) { return mat_vector_to_list(self.A); })
        .def_property_readonly("B", [](const Track& self) { return mat_vector_to_list(self.B); })
        .def_property_readonly("G", [](const Track& self) { return mat_to_numpy(self.G); })
        .def_property_readonly("Gi", [](const Track& self) { return mat_to_numpy(self.Gi); })
        .def_property_readonly("fi", [](const Track& self) { return mat_to_numpy(self.fi); })
        .def_property_readonly("Hi", [](const Track& self) { return mat_to_numpy(self.Hi); });
}

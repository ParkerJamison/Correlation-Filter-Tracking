
# Correlation Filter Tracker (CFT)

> Dual-filter MOSSE tracker with a native C++ implementation and Python bindings.

The project implements the MOSSE-based Correlation Filter Tracker described by Bolme *et al.* (CVPR 2010).  
Each object track carries both a long-term and a short-term filter. Response sharpness is measured through the
Peak-to-Sidelobe Ratio (PSR), allowing the tracker to decide which filters can be safely updated when the target
is partially occluded, leaving the frame, or changing appearance rapidly.

---

## Highlights
- Two-filter MOSSE tracker that keeps a stable long-term model while adapting quickly with a short-term model.
- Works in both C++ (`main.cpp`) and Python (`main.py`) through a shared pybind11 module (`cft_tracker`).
- Designed for live video sources (USB webcams, Raspberry Pi camera modules, recorded streams).
- Uses ROI selection (`cv::selectROI`) for intuitive initialization.
- Ships with simple build tooling: a `makefile` for the native tracker and `setup.py` for Python wheels.

---

## Repository Layout

| Path | Description |
| ---- | ----------- |
| `CFT_Track.hpp/cpp` | Core tracker implementation (learning, PSR logic, FFT helpers). |
| `TrackID.hpp/cpp` | Track state container (bounding boxes, masking/cropping helpers, filter storage). |
| `bindings.cpp` | pybind11 bindings that expose `CFT` and `Track` to Python. |
| `main.cpp` | Minimal C++ demo that runs the tracker on a webcam feed. |
| `main.py` | Python counterpart using the same tracker logic. |
| `setup.py` | Build script for the `cft_tracker` Python extension module. |
| `makefile` | Utility targets for compiling the native sample with OpenCV. |

---

## Requirements

| Component | Notes |
| --------- | ----- |
| Compiler | C++17-compliant toolchain (clang++ or g++). |
| OpenCV | Version 4.x with dev headers and `pkg-config` metadata. |
| Python | 3.9+ recommended for building the `cft_tracker` module. |
| Python packages | `pybind11`, `numpy`, `opencv-python` for runtime demos. |
| Misc | `pkg-config` must be able to locate `opencv4` (configure `PKG_CONFIG_PATH` if necessary). |

The ROI selection UI requires a GUI-enabled environment. Headless servers can initialize tracks by manually
setting bounding boxes inside `Track::initBBox`.

---

## Quick Start

### Python bindings

```bash
git clone https://github.com/ParkerJamison/CorrelationFilter.git
cd CorrelationFilter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pybind11 numpy opencv-python
python -m pip install -v .
python main.py          # launches the webcam demo
```

The `-v` flag on install is useful because OpenCV builds may take a moment and verbose logs help diagnose missing
include/lib paths. The build creates `cft_tracker.<platform>.so` in the project root when installed in-place.

### Native C++ sample

```bash
git clone https://github.com/ParkerJamison/CorrelationFilter.git
cd CorrelationFilter
make                # uses pkg-config opencv4 flags
./main              # starts the demo
```

The `makefile` defaults to the Homebrew OpenCV layout (`/opt/homebrew/Cellar/opencv/...`). Update `OPENCV_BASE`
or rely on the `pkg-config` flags exported by your installation if the defaults do not match your system.

---

## Usage

### Tracker lifecycle

1. Acquire an initial frame (`cv::VideoCapture`, GStreamer, recorded video, etc.).
2. Call `CFT::initTracking(frame)` to create a `Track`. The call displays `selectROI`, allowing you to draw the
   initial bounding box. The tracker computes an optimal padded FFT size and trains both filters.
3. For every subsequent frame, call `CFT::updateTracking(frame, track)`. The track maintains its own search area,
   gets cropped with replication padding, and the PSR-driven update rule decides which filters to refresh.
4. Render the latest bounding box through `track.getDisplayBBox()` (C++) or as a tuple in Python.

### Python example

```python
import cv2 as cv
import cft_tracker

cap = cv.VideoCapture(0)
ret, frame = cap.read()

tracker = cft_tracker.CFT()  # defaults: lr=0.125, sigma=100, numTrain=16
track = tracker.initTracking(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    tracker.updateTracking(frame, track)
    x, y, w, h = track.getDisplayBBox()
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if cv.waitKey(1) == ord("q"):
        break
```

See `main.py` for an extended version that shows FPS on-screen.

### C++ example

```cpp
CFT tracker;
std::vector<Track> tracks(1);
cv::Mat frame;
cv::VideoCapture cap(0);

cap >> frame;
tracks[0] = tracker.initTracking(frame);

while (cap.read(frame)) {
    for (auto &track : tracks) {
        tracker.updateTracking(frame, track);
        cv::rectangle(frame, track.getDisplayBBox(), cv::Scalar(255, 0, 0), 2);
    }
    cv::imshow("Tracker", frame);
    if (cv::waitKey(25) == 27) break;
}
```

This is essentially what `main.cpp` implements.

---

## Configuration

You can tune the tracker by selecting the appropriate `CFT` constructor overload.

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `lr` | `0.125` | Learning rate used for filter adaptation. Short-term filters internally blend with `0.15` to react faster. |
| `sigma` | `100` | Variance of the Gaussian response map. Larger values widen the desired peak. |
| `numTrain` | `16` | Number of random-warp augmentation steps when fitting the initial filter pair. |

`Track::psrFlag` toggles once either filter achieves a PSR greater than 13, preventing noisy early updates.

---

## Algorithm Overview

1. **Pre-processing (`preProcess`)**  
   Frames are converted to grayscale, log-scaled, mean/variance normalized, and tapered with a Hanning window to
   reduce boundary effects before entering the FFT domain.

2. **Filter training (`trainFilter`)**  
   The tracker synthesizes `numTrain` warped versions of the initial patch (random rotation and translation)
   to bootstrap both short- and long-term filters. Each warped sample is correlated with a Gaussian response map
   to build the numerator (`A`) and denominator (`B`) terms needed for the MOSSE solution.

3. **Update rule (`CFT::updateTracking`)**  
   Each frame generates two peak responses. The PSR is computed by masking a peak window and measuring how sharp
   the detection is relative to the sidelobes:  
   - **PSR ≥ 20**: confident detection → update both filters.  
   - **6 < PSR < 20**: partial occlusion or appearance shift → only update the short-term filter.  
   - **PSR ≤ 6**: unreliable detection → freeze both filters and keep searching.  

4. **Tracking outputs**  
   The best peak updates the bounding box center (`dx`, `dy`). The search region is clamped to image bounds and
   padded with replicated pixels whenever it drifts outside the frame.

---

## Development Tips
- Use `python -m pip install -e .` while iterating on the bindings to rebuild the extension in-place.
- If `pkg-config --cflags opencv4` fails, export `PKG_CONFIG_PATH` (e.g., `/opt/homebrew/lib/pkgconfig` on macOS).
- `make clean` removes object files and the `main` executable.
- Debug logging currently writes PSR values to stdout from `updateTracking`; redirect or remove for production.

---

## Roadmap
- **Add tracks in real time:** Buffer incoming frames while the user selects a new ROI, then fast-forward the new
  filter over the buffered frames to catch up with the live feed.
- **Graceful out-of-frame handling:** Enter a standby mode that expands the search window and polls less frequently
  until the target re-enters the frame or a timeout occurs.
- **Domain-specific modes:** Ship presets tuned for people, vehicles, drones, etc., to stabilize filter parameters
  based on expected motion/scale changes.

---

## References

- D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui,  
  "Visual Object Tracking using Adaptive Correlation Filters,"  
  *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2010.  
  [PDF](https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf)

---

## License

Distributed under the terms of the [GNU General Public License v3](LICENSE).

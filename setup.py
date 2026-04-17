from setuptools import setup, Extension
import pybind11
import subprocess
import shlex

def pkgconfig(package):
    """Helper to get compile/link flags from pkg-config."""
    try:
        output = subprocess.check_output(['pkg-config', '--cflags', '--libs', package]).decode('utf-8')
        flags = shlex.split(output)
        
        include_dirs = [f[2:] for f in flags if f.startswith('-I')]
        library_dirs = [f[2:] for f in flags if f.startswith('-L')]
        libraries = [f[2:] for f in flags if f.startswith('-l')]
        
        return {
            'include_dirs': include_dirs,
            'library_dirs': library_dirs,
            'libraries': libraries,
        }
    except subprocess.CalledProcessError:
        raise RuntimeError(f"pkg-config failed for {package}. Ensure OpenCV is installed and PKG_CONFIG_PATH is set.")

opencv_flags = pkgconfig('opencv4')

ext_modules = [
    Extension(
        'cft_tracker',
        sources=['CFT_Track.cpp', 'bindings.cpp', 'TrackID.cpp'],
        include_dirs=[pybind11.get_include()] + opencv_flags.get('include_dirs', []),
        library_dirs=opencv_flags.get('library_dirs', []),
        libraries=opencv_flags.get('libraries', []),
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='cft_tracker',
    ext_modules=ext_modules,
)

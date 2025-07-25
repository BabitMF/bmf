import os
import re
import site
import ctypes
import platform
import subprocess
import sys
import sysconfig
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# ref: https://github.com/pybind/cmake_example

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

package_version="0.2.0"

NAMESPACE = os.environ.get("BMF_PACKAGE_NAMESPACE", "")
PACKAGE_NAME = os.environ.get("BMF_PACKAGE_NAME_OVERRIDE", "BabitMF")
PACKAGE_VERSION = os.environ.get("BMF_VERSION_OVERRIDE", package_version)
PACKAGE_URL = os.environ.get("BMF_PACKAGE_URL_OVERRIDE", "https://github.com/BabitMF/BabitMF")
NAMESPACE_NESTING = len(NAMESPACE) > 0 and os.environ.get("BMF_ENABLE_NAMESPACE_NESTING", "OFF").upper() == "ON"

if "DEVICE" in os.environ and os.environ["DEVICE"] == "gpu":
    PACKAGE_NAME = PACKAGE_NAME + "_gpu"

PACKAGES = [
    'bmf',
    'bmf.builder',
    'bmf.cmd.python_wrapper',
    'bmf.ffmpeg_engine',
    'bmf.hmp',
    'bmf.modules',
    'bmf.python_sdk',
    'bmf.server',
    'bmf.templates',
]

PACKAGE_DATA = {
    "bmf.templates": ["jinja_templates/**/*.j2"]
}

CONSOLE_SCRIPTS = {
    'run_bmf_graph': 'bmf.cmd.python_wrapper.wrapper:run_bmf_graph',
    'trace_format_log': 'bmf.cmd.python_wrapper.wrapper:trace_format_log',
    'module_manager': 'bmf.cmd.python_wrapper.wrapper:module_manager',
    'bmf_env': 'bmf.cmd.python_wrapper.wrapper:bmf_env',
    'bmf_template_generator': 'bmf.templates.cli:main',
}

if NAMESPACE_NESTING:
    PACKAGES = [NAMESPACE] + [NAMESPACE + "." + p for p in PACKAGES]
    PACKAGE_DATA = {NAMESPACE + "." + k: v for k, v in PACKAGE_DATA.items()}
    CONSOLE_SCRIPTS = [f"{k} = {NAMESPACE}.{v}" for k, v in CONSOLE_SCRIPTS.items()]
else:
    CONSOLE_SCRIPTS = [f"{k} = {v}" for k, v in CONSOLE_SCRIPTS.items()]

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        sha = os.environ.get("GIT_SHA")
        short_sha = sha[:7]

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DBMF_ENABLE_TEST=OFF",
            f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DPYTHON_LIBRARY={sysconfig.get_config_var('LIBDIR')}",
            f"-DBMF_PYENV={'{}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)}", #There is a situation on macOS, that is, cibuildwheel uses python3.8.10 to call the setup.py, but cmake finds python3.8.9 installed under the xcode path, resulting in the python path in the final executable file starting with @rpath, which is There will be problems at runtime. So we use the full python version number, which is major.minor.patch
            f"-DBMF_BUILD_VERSION={PACKAGE_VERSION}",
            f"-DBMF_BUILD_COMMIT={short_sha}",
        ]

        if debug:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]
            if debug == 2:
                cmake_args += ["--trace-expand"]

        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}{os.sep}lib"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("running cmake configure:{} {} {}, cwd:{}".format("cmake", ext.sourcedir, cmake_args, build_temp))
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        print("running cmake build:{} {} {} {}, cwd:{}".format("cmake", "--build", ".", build_args, build_temp))
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

        # This CMake building support only single output, and scikit-build using pyproject.toml which is
        # a static config file. Obviously, setup.py is more convenient than pyproject.toml, so we manually copy
        # _bmf, _hmp, py_module_loader, go_module_loader and builtin_moduls before repair, instead of
        # the entire lib directory. the build directory is temporary, so we need to copy it here, instead of package_data.
        output_prefix = os.path.join(extdir, NAMESPACE) if NAMESPACE_NESTING else extdir
        bmf_output_root = os.path.join(output_prefix, "bmf")
        for output_dir in ["bin", "lib", "include", "cpp_modules", "python_modules"]: #"go_modules"
            shutil.copytree(os.path.join(build_temp, "output", "bmf", output_dir), os.path.join(bmf_output_root, output_dir))
        for file in ["BUILTIN_CONFIG.json"]:
           shutil.copyfile(os.path.join(build_temp, "output", "bmf", file), os.path.join(bmf_output_root, file))

        # TODO: Remove versioned libraries to reduce package size

        if sys.platform.startswith("darwin"):
            subprocess.run(
                ["./scripts/redirect_macos_dep.sh", output_prefix], check=True
            )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name=NAMESPACE + "." + PACKAGE_NAME if NAMESPACE else PACKAGE_NAME,
    version=PACKAGE_VERSION,
    author="",
    author_email="",
    python_requires='>= 3.6',
    description="Babit Multimedia Framework",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    readme='README.md',
    url=PACKAGE_URL,
    install_requires=[
        "numpy>=1.19.5,<2.0.0",
    ],
    extras_require={
        "test": ["pytest>=6.2.5,<=8.4.1"],
        "templates": [
            "jinja2>=2.11.0,<=3.1.6",
            "click>=7.1.2,<=8.2.1",
            "InquirerPy>=0.0.1,<=0.3.4",
        ],
    },
    package_data=PACKAGE_DATA,
    packages=PACKAGES,
    ext_modules=[CMakeExtension("bmf")],
    cmdclass={"build_ext": CMakeBuild},
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    }
)

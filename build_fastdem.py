"""Build the robot's pinned FastDEM adapter locally, with a portable Windows compiler.

uv run --no-sync --with cmake --with ninja python build_fastdem.py --llvm-mingw --robot-repo PATH
All downloaded source, licenses and build products stay under logs/fastdem.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

FASTDEM_REVISION = "5cd3e2552dba62be14cd8fe83d5c81261eafd12e"
NANOGRID_REVISION = "058a6a60aa5c4c681be3428f57dbfc5b8fd57f96"
LLVM_MINGW_URL = "https://github.com/mstorsjo/llvm-mingw/releases/download/20261117/llvm-mingw-20261117-ucrt-x86_64.zip"
LLVM_MINGW_SHA256 = "4d002e330a0f703625aa5ccbc37e96f8f824413e163bfc7336774aabd2c3a412"
SOURCES = {
    "eigen": "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip",
    "yaml": "https://codeload.github.com/jbeder/yaml-cpp/zip/refs/tags/0.8.0",
    "spdlog": "https://codeload.github.com/gabime/spdlog/zip/refs/tags/v1.15.3",
    "nanogrid": f"https://codeload.github.com/Ikhyeon-Cho/nanoGrid/zip/{NANOGRID_REVISION}",
}


def run(*command, cwd=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def download_source(work, name, url, expected_sha256=None):
    archive = work / f"{name}.zip"
    if not archive.is_file():
        print(f"Downloading {name}: {url}", flush=True)
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Checksum mismatch for {archive}")
    with zipfile.ZipFile(archive) as package:
        root = package.namelist()[0].split("/")[0]
        destination = work / root
        if not destination.is_dir():
            for entry in package.namelist():
                (work / entry).resolve().relative_to(work)
            package.extractall(work)
    return destination, {"url": url, "sha256": digest}


def build(args):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    robot = args.robot_repo.resolve()
    bench = robot / "native_stack/tools/heightmap_bench"
    patch = bench / "fastdem-core.patch"
    adapter = bench / "fastdem_replay.cpp"
    if not patch.is_file() or not adapter.is_file():
        raise FileNotFoundError("Pull kaivuriprokkis main before building")
    core = work / "FastDEM"
    if not core.is_dir():
        run("git", "clone", "https://github.com/Ikhyeon-Cho/FastDEM.git", core)
        run("git", "checkout", FASTDEM_REVISION, cwd=core)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip()
    if revision != FASTDEM_REVISION:
        raise ValueError(f"Expected FastDEM {FASTDEM_REVISION}; found {revision} in {core}")
    if subprocess.run(["git", "apply", "--check", str(patch)], cwd=core, capture_output=True).returncode == 0:
        run("git", "apply", patch, cwd=core)
    else:
        run("git", "apply", "--reverse", "--check", patch, cwd=core)
    paths, manifest = {}, {"fastdem_revision": revision, "nanogrid_revision": NANOGRID_REVISION}
    for name, url in SOURCES.items():
        paths[name], manifest[name] = download_source(work, name, url)
    source = work / "project"
    source.mkdir(exist_ok=True)
    # Preserve the robot adapter verbatim except Windows binary pipe setup.
    cpp = adapter.read_text(encoding="utf-8")
    if cpp.count("int main(int argc, char** argv) {") != 1:
        raise ValueError("The robot adapter entry point changed; review the Windows binary pipe setup")
    cpp = cpp.replace(
        "int main(int argc, char** argv) {",
        "#ifdef _WIN32\n#include <fcntl.h>\n#include <io.h>\n#endif\n\n"
        "int main(int argc, char** argv) {\n#ifdef _WIN32\n"
        "  _setmode(_fileno(stdin), _O_BINARY);\n"
        "  _setmode(_fileno(stdout), _O_BINARY);\n#endif",
    )
    (source / "fastdem_replay.cpp").write_text(cpp, encoding="utf-8")
    manifest["adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    manifest["core_patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
    manifest["tool_versions"] = {
        package: importlib.metadata.version(package)
        for package in ("cmake", "ninja")
        if importlib.util.find_spec(package) is not None
    }
    cmake = """cmake_minimum_required(VERSION 3.20)
project(robot_fastdem LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "@eigen@")
add_subdirectory("@yaml@" yaml)
add_subdirectory("@spdlog@" spdlog)
# fmt bundled by spdlog 1.15 needs C allocation declarations with newer libc++.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(spdlog PUBLIC -include stdlib.h)
endif()
file(GLOB NG_SOURCES "@nanogrid@/src/*.cpp")
add_library(nanoGrid STATIC ${NG_SOURCES})
target_include_directories(nanoGrid PUBLIC "@nanogrid@/include")
target_link_libraries(nanoGrid PUBLIC Eigen3::Eigen)
file(GLOB FD_SOURCES "@core@/fastdem/src/*.cpp")
add_library(fastdem_core STATIC ${FD_SOURCES})
target_include_directories(fastdem_core PUBLIC
  "@core@/fastdem/include" "@core@/fastdem/lib"
  "@core@/fastdem/lib/nanoPCL/include" "@core@/fastdem/lib/nanoPCL/thirdparty"
  "@core@/fastdem/lib/stb")
target_link_libraries(fastdem_core PUBLIC nanoGrid yaml-cpp spdlog::spdlog)
add_executable(fastdem_replay fastdem_replay.cpp)
target_link_libraries(fastdem_replay PRIVATE fastdem_core)
if(MSVC)
  target_compile_options(fastdem_core PRIVATE /bigobj)
endif()
"""
    for name, path in {**paths, "core": core}.items():
        cmake = cmake.replace(f"@{name}@", path.as_posix())
    (source / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    build_dir = work / ("build-mingw" if args.llvm_mingw else "build")
    configure = ["cmake", "-S", source, "-B", build_dir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"]
    if args.llvm_mingw:
        if os.name != "nt":
            raise ValueError("The bundled LLVM-MinGW toolchain targets Windows x64")
        compiler, manifest["llvm_mingw"] = download_source(work, "llvm-mingw", LLVM_MINGW_URL, LLVM_MINGW_SHA256)
        binary = compiler / "bin"
        os.environ["PATH"] = str(binary) + os.pathsep + os.environ["PATH"]
        configure += [
            f"-DCMAKE_C_COMPILER={binary / 'x86_64-w64-mingw32-clang.exe'}",
            f"-DCMAKE_CXX_COMPILER={binary / 'x86_64-w64-mingw32-clang++.exe'}",
            "-DCMAKE_EXE_LINKER_FLAGS=-static",
        ]
    run(*configure)
    run("cmake", "--build", build_dir, "--parallel", args.jobs)
    (work / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Built:", build_dir / ("fastdem_replay.exe" if os.name == "nt" else "fastdem_replay"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("logs/fastdem"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--llvm-mingw", action="store_true", help="Download a portable Windows C++ toolchain; no admin required"
    )
    args = parser.parse_args()
    if not shutil.which("cmake") or not shutil.which("ninja"):
        parser.error("CMake and Ninja are required; see the uv command in this script's docstring")
    build(args)


if __name__ == "__main__":
    main()

from setuptools import setup

setup(
    name="bmf",
    version="0.1.0",
    python_requires='>= 3.6',
    description="Babit Multimedia Framework",
    url="www.bytedance.com",
    install_requires=[
        "numpy >= 1.19.5"
    ],
    packages=[
        'bmf',
        'bmf.builder',
        'bmf.ffmpeg_engine',
        'bmf.python_sdk',
        'bmf.modules',
        'bmf.server',
        'bmf.lib',
        'bmf.bin',
        'bmf.include',
    ],
    package_dir={
        'bmf': 'bmf',
        'bmf.builder': 'bmf/builder',
        'bmf.ffmpeg_engine': 'bmf/ffmpeg_engine',
        'bmf.python_sdk': 'bmf/python_sdk',
        'bmf.modules': 'bmf/modules',
        'bmf.server': 'bmf/server',
        'bmf.lib': 'bmf/lib',
        'bmf.bin': 'bmf/bin',
        'bmf.include': 'bmf/include',
    },
    include_package_data=True,
)

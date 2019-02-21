from setuptools import setup, find_packages

setup(
    name="flopth",
    version="0.0.1",
    keywords=("flopth", "Pytorch", 'Flops', 'Deep-learning'),
    description="A program to calculate FLOPs of Pytorch models",
    long_description="flopth is a program to calculate the FLOPs of Pytorch models, with cli tool and Python API. Support multiple kinds of input, support CPU and GPU, support extra parameters in forward function",
    license="MIT Licence",

    url="https://github.com/vra/flopth",
    author="Yunfeng Wang",
    author_email="wyf.brz@gmail.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[],

    scripts=[],
    entry_points={
        'console_scripts': [
            'flopth=flopth.__init__:main'
        ]
    }
)

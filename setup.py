from setuptools import setup, find_packages

readme = open('README.md').read()
#version = open('./version.txt', 'r').read().strip()
#print(version)

setup(
    name="flopth",
    version='0.1.3',
    keywords=("flopth", "Pytorch", "Flops", "Deep-learning"),
    description="A program to calculate FLOPs and Parameters of Pytorch models",
    long_description=readme,
    long_description_content_type="text/markdown",

    license="MIT Licence",
    url="https://github.com/vra/flopth",
    author="Yunfeng Wang",
    author_email="wyf.brz@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "numpy",
        "tabulate",
        "torch",
        "torchvision",
    ],
    scripts=[],
    entry_points={"console_scripts": ["flopth=flopth.__init__:main"]},
)

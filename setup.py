from setuptools import setup

setup(
    name="direct_maxflux",
    version="0.1.0",
    py_modules=["dmf"],
    install_requires=[
        'ase',
        'cyipopt',
    ],
    python_requires=">=3.6",
)

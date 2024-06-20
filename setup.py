import setuptools
import os

# read content of README.md
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="img_quality_cil_stir",
    use_scm_version={'fallback_version':'unkown'},
    setup_requires=['setuptools_scm','setuptools_scm_git_archive'],
    author="Georg Schramm, Junqi Tang, Imraj Singh",
    author_email="georg.schramm@kuleuven.be",
    description="minimal python package for an image quality callback for CIL and STIR images",
    long_description=long_description,
    license='Apache License 2.0',
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=['numpy >= 1.0',
                      'scipy >= 1.0',
                      'pandas >= 1.0',
                      'tensorboard >= 2.0',
                      'tensorboardx >= 2.0'],
    packages=['img_quality_cil_stir'],
)

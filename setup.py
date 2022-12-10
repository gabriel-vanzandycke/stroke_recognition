from setuptools import setup, find_packages

setup(
    name='stroke_recognition',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/stroke_recognition",
    licence="LGPL",
    python_requires='>=3.8',
    description="Exercice for SONY Depth Sensing interview",
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "calib3d",
        "jupyter",
        "scikit-learn",
        "cv2",
        "imageio",

    ],
)

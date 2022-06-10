from setuptools import setup, find_packages

setup(
    name='squeezeformer',
    packages=find_packages(),
    version='0.1.0',
    description='An Efficient Transformer for Automatic Speech Recognition',
    author='Ha Sangchun',
    author_email='seomk9896@gmail.com',
    url='https://github.com/upskyy/Squeezeformer',
    keywords=['asr', 'speech_recognition', 'artificial intelligence', 'transformer'],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.4.0',
        'numpy',
    ],
)

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='squeezeformer',
    packages=find_packages(),
    version='0.1.1',
    description='An Efficient Transformer for Automatic Speech Recognition',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ha Sangchun',
    author_email='seomk9896@gmail.com',
    url='https://github.com/upskyy/Squeezeformer',
    keywords=['asr', 'speech_recognition', 'artificial intelligence'],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.4.0',
        'numpy',
    ],
)

from setuptools import setup, find_packages

setup(
    name='src',
    version='0.1.0',  # Your package version
    description='source code for number encoding in LLMs',
    author='Tung Nguyen',
    author_email='ductungnguyen1997@gmail.com',
    # url='https://github.com/tung-nd/stormer-dev',  # Optional
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        # Choose your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
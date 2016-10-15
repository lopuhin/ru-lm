from setuptools import setup, find_packages


setup(
    name='ru-lm',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
    ],
    entry_points={
        'console_scripts': [
            'rnn-word-lm=rnn.word_lm:run',
        ],
    },
)

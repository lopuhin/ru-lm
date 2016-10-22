from setuptools import setup, find_packages


setup(
    name='ru-lm',
    packages=find_packages(),
    install_requires=[
        'flask',
        'numpy',
        'progressbar2',
        'tensorflow',
    ],
    entry_points={
        'console_scripts': [
            'rnn-word-lm=rnn.word_lm:run',
            'rnn-web=rnn.web:run',
        ],
    },
)

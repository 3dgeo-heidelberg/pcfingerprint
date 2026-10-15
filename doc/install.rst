Installation
============

``pcfingerprint`` does not yet have an official release, but can be installed from source.
``pcfingerprint`` requires Python >= 3.10. We recommend using a virtual environment to manage dependencies, e.g., `conda`/`mamba`::

    mamba create -n pcfingerprint python=3.13
    mamba activate pcfingerprint

The following sequence of commands is used to install ``pcfingerprint`` from source using pip. It is recommended to do this in a virtual environment::

    git clone https://github.com/3dgeo-heidelberg/pcfingerprint.git
    cd pcfingerprint
    python -m pip install -v --editable .

The ``--editable``  flag allows you to change the Python sources of ``pcfingerprint``  without reinstalling the package.
The ``-v`` flag enables verbose output which gives you detailed information about the compilation process that you should include into potential bug reports.

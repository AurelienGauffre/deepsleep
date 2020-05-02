# How to contribute
## Getting started
[![Python version](https://img.shields.io/badge/python-3.6.9%2B-blue)](https://www.python.org/downloads/release/python-369/)

The project was written using **Python 3.6.9+** and uses [**Poetry**](https://python-poetry.org/) as a dependency manager.

:open_file_folder: **Clone the repository**  
Clone the repository and dive into it:
```shell
git clone https://github.com/AurelienGauffre/DeepSleep
cd DeepSleep
```

:four_leaf_clover: **Install poetry**  
[**Poetry**](https://python-poetry.org/) is a dependency management tool that 
we use a lot throughout our development process. You can install Poetry
isolated from the rest of your system (recommended way) by following the 
instructions [here](https://python-poetry.org/docs/#installation).

:books: **Install the project dependencies**  
Many libraries are used by the project and for development (a complete list of 
dependencies can be found in [`pyproject.toml`](pyproject.toml)). Run the 
following from within the project folder to install all the dependencies:
```shell
poetry install
```

:warning: For Windows user, you may also run the following command to
install PyTorch after you've successfully installed the other dependencies
using the previous command:
```shell
poetry run pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Write some code
This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.

:point_up: A few notes:
- We recommend to set your IDE line length ruler to 79 characters.
- Please use `'` rather than `"` for strings:
  ```diff
  - message = "Hello world!"
  + message = 'Hello world!'
  ```
- Please use no capital letter or point in short inline comments:
  ```diff
  - # Let's add the git submodule to our Python path.
  + # let's add the git submodule to our Python path
  ```

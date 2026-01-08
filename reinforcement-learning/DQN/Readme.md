# DRL DQN

### Topics
- DQN
- Rainbow
    - Double Q-learning
    - Prioritized Replay
    - Dueling Network
    - Multi-step Learning
    - Distributional RL
    - Noisy Nets

### Structure

Follow the "dqn.ipynb" notebook for instructions:


You can run the **optional** test script that covers replay buffers from the "test" directory using:

```
python -m unittest test_replaybuffer.py
```

### Installation

You need to install requirements. I recommend that you use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment for this project.

```
conda create -n dqn python=3.7
conda activate dqn
```

If you are going to use GPU, install [Pytorch](https://pytorch.org/get-started/locally/) using the link and remove it from requirements.

You can install the requirements with the following commands in the homework directory:

```
conda install -c conda-forge swig
conda install nodejs
pip install -r requirements.txt
python -m ipykernel install --user --name=dqn
```
Then you need to install the project package. You can install the package with the following command: (Make sure that you are at the project directory.)

```
pip install -e .
```

This command will install the project package in development mode so that the installation location will be the current directory.

### Related Readings

- [DQN](https://www.nature.com/articles/nature14236)
- [Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
- [Prioritized Replay](https://arxiv.org/pdf/1511.05952.pdf)
- [Dueling Network](https://arxiv.org/pdf/1511.06581.pdf)
- Multi-step Learning - Richard S. Sutton and Andrew G. Barto Chapter 7
- [Distributional RL](https://arxiv.org/pdf/1707.06887.pdf)
- [Noisy Nets](https://arxiv.org/pdf/1706.10295.pdf)
- [Rainbow](https://arxiv.org/pdf/1710.02298.pdf)

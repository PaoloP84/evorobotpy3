# evorobotpy3
Evorobotpy3 is a flexible and easy-to-use simulation tool allowing users to implement and test algorithms. The tool provides a list of benchmark problems like classic control, robot locomotion, swarm robotics, competitive co-evolution and others. Evorobotpy3 contains a lot of algorithms ranging from classic Evolutionary Algorithms (e.g., a basic version of the Genetic Algorithm, the Generational Genetic Algorithm, the Steady-State algorithm) to modern Evolutionary Strategies (e.g., CMA-ES, OpenAI-ES, xNES), and supports the use of RL methods like A2C, DDPG, PPO, etc. In addition, the users can define controllers through different types of neural networks (feed-forward, recurrent, LSTM).

One of the most important features of evorobotpy3 is its usability: users can run experiments by specifying all the parameters in a configuration file (see file config.ini in the homepage of the repo). The file is organized in four main sections:

1) [EXP]: this section contains the definition of the environment and the algorithm to be used;
2) [ALGO]: this section includes the parameters related to the specific algorithm indicated in the previous section;
3) [POLICY]: this section contains information about the number of agents, the evaluation of each agent and the neural network controller used;
4) [ENV]: this section is optional and allows to specify optional parameters for the environment.

It is worth noting that training RL algorithms follows a different approach: the configuration file used to train RL algorithms contains only a mandatory [EXP] section, in which the parameter environment must be necessarily specified. Moreover, the file could contain the optional [ENV] section in order to specify the parameters for the task. The readers can have a look at the config.ini file included in the baselines sub-folder. 

Evorobotpy3 has been presented during the 2025 Genetic and Evolutionary Computation Conference (GECCO 2025). You can find the paper at https://dl.acm.org/doi/10.1145/3712255.3726545

Please use this BibTeX to cite this repository in your publications:
```
@inproceedings{evorobotpy3,
  author = {Pagliuca, Paolo and Nolfi, Stefano and Vitanza, Alessandra},
  title = {Evorobotpy3: a flexible and easy-to-use simulation tool for Evolutionary Robotics},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO 2025 Companion)},
  year = {2025}
}
```

The authors want to thank Pedro Neves for the implementation of the Double T-Maze environment.

# gym-dotsboxes
Dots and Boxes in Python Gym with two Dyna-Q+ Agents

How to run:
Will not work for Python 3.8 or any versions below Python 3.5. Make sure the package gym is installed, then do the following in cmd:

cd gym-dotsboxes

pip install -e .


To run a game between two agents, please run basic_agent.py.

Current Project:
- Dots and Boxes Gym environment that is a set size of 4x4 dots.
- Two random agents playing within environment.

Future Changes:
- Add Dyna-Q+ agents trained in various ways (i.e. epsilon-greedy, temporal difference).
- Add ability to dynamically change environment, such as board size.
- Changes to interface, add GUI that may allow a human player to operate in place of one agent.

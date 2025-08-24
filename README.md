# Antistalling pivot rule
File `TetsPivots_final.py` contains an implementation of the [Antistalling pivot rule](https://link.springer.com/chapter/10.1007/978-3-031-59835-7_19) using [CyLP](https://github.com/coin-or/CyLP).  
The code allows to test the pivot rule agains others, such as Dantzig's, Steepest edge, Bland's, LIFO, Most frequent, [Positive edge](https://www.gerad.ca/en/papers/G-2010-61.pdf).  
Implementations of many aforemention rules were adapted from https://coin-or.github.io/CyLP/modules/pivots.html .  

## Usage:
Change `foldername` variable int the `__main__` function to the relative path to your folder with .mps problem files  
Variable `outputFile` specifies the name of the csv file with results. It will contain the total number of pivots of each pivor rule for theach file, along with the total number of degenerate pivots.

## Dependencies:
[tdqm](https://pypi.org/project/tdqm/) for progress bar  
[CyLP](https://github.com/coin-or/CyLP) for implemeting pivot rules

__Warning__: current CyLP release [v0.93.1](https://github.com/coin-or/CyLP/releases/tag/v0.93.1) has an open [issue](https://github.com/coin-or/CyLP/issues/105) that dissallows the usage of your own implementations of pivot rules. Therefore one has to use versions before commit [`72d66b5`](https://github.com/coin-or/CyLP/commit/72d66b58af5ac0cee25d94b63115c6f65e3cff8b). We only managed to install a correspoding version of CyLP from source.



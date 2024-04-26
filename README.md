# Machine Vision Toolkit

MVT (Machine Vision Toolkit) is a collection of utilities and scripts to facilitate the training of machine learning models and write computer vision algorithms. Among others, the code allows to create image classifiers, feature extractors, data generators, train deep neural networks and plot graphs.

## Installation

Create a new conda environment and install the required packages.

```bash
conda create -n mvt
conda activate mvt
conda install python=3.7

pip install .
```

If you plan to modify the code, use `-e` flag to install the package in editable mode.  
Add `--force` flag to force the setup installation if required modules have changed.

```bash
pip install -e . --force
```

## Notes

Writing styles and conventions are based on [this Python guide](https://docs.python-guide.org/writing/structure/).

## License

[Machine Vision Toolkit](https://github.com/mcarletti/machinevisiontoolkit) © 2024 by [Marco Carletti](https://www.marcocarletti.it/) is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/?ref=chooser-v1).

CC BY-SA 4.0  
**Attribution-ShareAlike 4.0 International**  
This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, even for commercial purposes. If others remix, adapt, or build upon the material, they must license the modified material under identical terms.

BY: Credit must be given to you, the creator.  
SA: Adaptations must be shared under the same terms.

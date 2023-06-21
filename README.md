# orchestra-jh

This is a fork of [Orchestra](https://github.com/AbdallahHemdan/Orchestra), a python package for OMR (opetical music recognition) that converts to [GUIDO musical notation](https://en.wikipedia.org/wiki/GUIDO_music_notation#:~:text=GUIDO%20Music%20Notation%20is%20a,musical%20notation%201%2C000%20years%20ago) by [Abdallah Hemdan](https://github.com/AbdallahHemdan), [Adel Rizq](https://github.com/AdelRizq), [Ahmed Mahboub](https://github.com/Mahboub99) and [
Kareem Mohamed](https://github.com/kareem3m) (available under [MIT License](https://github.com/AbdallahHemdan/Orchestra/blob/master/LICENSE)).

## Changes:

- Wrapped into a package
- Exporting more data (export coordinates of each event).

## Usage:

You'll need to `pip install` the package.

### Recognition

```python
import orchestrajh as oc

oc.process(
    "/Input", # A folder of images
    "/Output", # An output folder
    filename = "/Users/jacob/Documents/Git Repos/orchestra-jh/model/model.sav" # the model, download here: https://github.com/AbdallahHemdan/Orchestra/tree/master/model (model.sav)
)
```
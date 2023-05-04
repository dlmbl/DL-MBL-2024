# DL@MBL 2023 Exercises

This repository contains the exercises for the "Deep Learning for Microscopy
Image Analysis" 2023 course at the MBL in Woods Hole.

This is a "super repository", containing the actual exercise repositories as
git submodules. To see the exercises, run the following after cloning this
repository:

```
git pull
git submodule update --init
```

This will pull in all the exercises. If you want to see only a specific
exercise, type

```
git pull
git submodule update --init <path>
```
where `<path>` is the name of the exercise (e.g., `02_intro_dl`).

## Instructions

Each exercise has its own subdirectory. In it, you will find a short `README`
with instructions how to set up the `conda` environment for this exercise and
get started with the exercise Jupyter notebook. All other instructions and task
descriptions are contained in the notebook.

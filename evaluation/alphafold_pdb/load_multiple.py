#! /usr/bin/env python3
from pymol import cmd
from glob import glob
#/path/**/*.c', recursive=True
lst = glob("**/*.pdb", recursive=True)
lst.sort()
for fil in lst: cmd.load(fil)
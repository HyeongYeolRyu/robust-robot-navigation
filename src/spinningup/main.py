import sys

print('sys.path from main.py')
print('\n'.join(sys.path))
print(__package__)
print(__name__)


from spinup import lib

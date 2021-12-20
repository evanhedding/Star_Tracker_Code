#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math as m
import numpy as np

#Incoming is degrees

def rotx(a):
    a = np.radians(a)
    Rx = np.array([[1,        0,         0],
                  [0, m.cos(a), -m.sin(a)],
                  [0, m.sin(a),  m.cos(a)]])
    return Rx
    
def roty(a):
    a = np.radians(a)
    Ry = np.array([[m.cos(a),  0,  m.sin(a)],
                    [0,         1,         0],
                    [-m.sin(a), 0,  m.cos(a)]])
    return Ry

def rotz(a):
    a = np.radians(a)
    Rz = np.array([[m.cos(a), -m.sin(a), 0],
                   [m.sin(a), m.cos(a), 0],
                   [0,         0,       1]])
    return Rz

def transform(x,y,z):
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])
    return T

def flipy(a):
    a = np.radians(a)
    Ry = np.array([[m.cos(a),  0,  m.sin(a), 0],
                    [0,         1,         0, 0],
                    [-m.sin(a), 0,  m.cos(a), 0],
                    [0,         0,         0, 1]])
    return Ry

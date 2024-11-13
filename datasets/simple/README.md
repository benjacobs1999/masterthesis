### This is all taken from https://github.com/locuslab/DC3/blob/main/datasets/simple/make_dataset.py to generate a simple QP problem dataset.

minimize_y 1/2 * y^T Q y + p^Ty
s.t.       Ay =  x
           Gy <= h




import cupy as cp

x = cp.arange(10)
print("x on GPU:", x)
print("x squared on GPU:", x**2)
print("Is the default device GPU?", cp.cuda.Device())



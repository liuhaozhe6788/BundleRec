def rel_impr(model, base):
    return ((model-0.5)/(base-0.5)-1) * 100
print(rel_impr(60.38, 59.39))
print(rel_impr(60.11, 59.39))
print(rel_impr(59.86, 59.39))
print(rel_impr(80.48, 79.39))
print(rel_impr(80.22, 79.39))
print(rel_impr(79.45, 79.39))
print(rel_impr(84.51, 83.37))
print(rel_impr(81.38, 83.37))
print(rel_impr(82.61, 83.37))
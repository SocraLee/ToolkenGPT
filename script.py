import re
s = '456abc123abc'
q = '123abc456abc'

# def natural_sort_key(s):
#     return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)][-2]
# c =[q,s]
# c.sort(key=natural_sort_key)
# print(c)

a = [-5,-4,-3,-2,-1]
print(a[:-2])
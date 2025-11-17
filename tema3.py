    
def prog_aritm(n):
    if n == 0:
        return 2
    else:
        return 2 * prog_aritm(n - 1) - 3


print(prog_aritm(8))


def fibo(n):
    if n == 1:
        return 1
    elif n == 0:
        return 0
    else:
        return fibo(n - 1) + fibo(n - 2)


print(fibo(6))


def suma(n):
    if n == 0:
        return 0
    elif n > 0:
        return suma(n - 1) + n


print(suma(10))


def prod_cif(n):
    if n < 10:
        return n
    else:
        return (n % 10) * prod_cif(n // 10)


print(prod_cif(82736))


def nr_cif(n):
    if n < 10:
        return 1
    else:
        return nr_cif(n // 10) + 1


print(nr_cif(82736))


def max_cif(n):
    if n < 10:
        return n
    else:
        return max(n % 10, max_cif(n // 10))


print(max_cif(42164))


def nr_cif_par(n):
    if n < 10:
        return 1 if (n % 2 == 0) else 0
    elif n % 2 == 0:
        return nr_cif_par(n // 10) + 1
    else:
        return nr_cif_par(n // 10)


print(nr_cif_par(82836))


def putere(a, n):
    if n == 0:
        return 1
    else:
        return a * putere(a, n - 1)


print(putere(10, 4))


def num_prim(n, d=2):
    if n < 2:
        return False
    if n == d:
        return True
    elif n % d == 0:
        return False
    else:
        return num_prim(n, d + 1)


print(num_prim(24))


def cmmdc(a, b):
    if b == 0:
        return a
    else:
        return cmmdc(b, a % b)


print(cmmdc(24, 36))


def reverse(s):
    if len(s) <= 1:
        return s
    return reverse(s[1:]) + s[0]


text = "portocale"
print(reverse(text))


def interval(start, end):
    if start == end:
        return end
    else:
        print(start)
        return interval(start + 1, end)


print(interval(12, 17))


def aparitii(n, cif):
    if n < 10:
        return 1 if cif == n else 0
    else:
        if cif == (n % 10):
            return 1 + aparitii(n // 10, cif)
        else:
            return aparitii(n // 10, cif)


print(aparitii(242344, 4))


def palindrom(n, ogl=0, aux=None):
    if aux is None:
        aux = n
    if n == 0:
        return ogl == aux
    ogl = ogl * 10 + (n % 10)
    return palindrom(n // 10, ogl, aux)


print(palindrom(22322))


def comp_func(f, x, n):
    if n == 0:
        return f(x)
    else:
        return f(x) + comp_func(f, x, n - 1)


print(comp_func(suma, 10, 5))


def suma_i(n):
    if n == 0:
        return 1
    else:
        return 1 / n + 1 + suma_i(n - 1)


print(suma_i(9))


import math

def suma_taylor(n, x):
    if n == 0:
        return 1
    else:
        return x * x / math.factorial(n) + suma_taylor(n - 1, x * x)

print(suma_taylor(5, 1.0))


def conv_binar(n):
    if (n == 1):
        return n % 2
    else:
        return (conv_binar(int(n / 2))) * 10 + n % 2

print(conv_binar(17))

def triunghi(n, x=1):
    if n == 0:
        return
    if x <= n:
        triunghi(n, x + 1)
        print(n, end = " ")
    else:
        triunghi(n - 1, 1)
        print()
triunghi(5)


def putere_mod(a, p):
    if a % p == 0:
        return 0
    def cauta(k, valoare):
        if valoare == 1:
            return k
        return cauta(k + 1, (valoare * a) % p)
    return cauta(1, a % p)

print()
print(putere_mod(3, 7))  

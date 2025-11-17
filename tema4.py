#1
rezultat = list(filter(lambda x : x % 2 == 0, [1, 2, 3, 4, 5, 6, 76]))
print(rezultat)

#2
sir_de_char = ["mar", "par", "apple", "orange"]
rez_filter_char = list(filter(lambda x: len(x) <= 3, sir_de_char))
print(rez_filter_char)

#3
""""
test = ["mar", 1, 2, 0, 'c']
print(test.sort())
"""

#4
sortare_lista = [1 , 2, 3, 9, 420, 67]
sortare_lista.sort(key = lambda x: x % 10)
print(sortare_lista)

#5
import functools, operator
numere_reale = [1.23, 2324.23, 324.3, 1324.2]
suma = functools.reduce(operator.add, numere_reale)
print(suma)

#6
rezultat_putere = list(map(lambda x: x ** 3, sortare_lista))
print(rezultat_putere)

#7
def verif_prim(x):
    if (x == 0 or x == 1):
        return 0
    if (x == 2):
        return 1
    d = 2
    while (d * d <= x):
        if (x % d == 0):
            return 0
        d = d + 1
    return 1
prime = list(filter(verif_prim, sortare_lista))
print(prime)
'''
#part2

#1
def creare_lista(n, lista):
    if (n < 10):
        if (verif_prim(n) == 1):
            lista.append(n)
        return lista
    else:
        if (verif_prim(n % 10) == 1):
            lista.append(n % 10)
        return creare_lista(int (n / 10), lista)

print(creare_lista(12475, []))
lista_smec = creare_lista(12475, [])


def creare_numar(n, lista):
    i = 0
    while (i < len(lista)):
        n = n * 10 + lista[i]
        i = i + 1
    return n
print(creare_numar(0, lista_smec))

def creare_numar_recursiv(n, lista, i):
    if (i == len(lista) - 1):
        n = n * 10 + lista[i]
        return n
    else:
        n = n * 10 + lista[i]
        return creare_numar_recursiv(n, lista, i + 1)
print(creare_numar_recursiv(0, lista_smec, 0))

def fromto(a, b, c, lista):
    i = a
    while (i <= b):
        if i % c == 0:
            lista.append(i)
        i = i + 1
    return lista
print(fromto(10, 20, 3, []))

def nth (n, lista):
    i = 0
    while (i < n - 1):
        i+=1
    return lista[i]

print(nth(4, [1,2,3,4,5,6,9]))

def firstn(n, lista):
    i = 0
    while (i < n):
        print(lista[i])
        i+=1
print(firstn(4, [1,24,52,5,45,42]))

def split(pairs):
    a = []
    b = []
    for p in pairs:
        try:
            x, y = p
        except Exception:
            continue
        a.append(x)
        b.append(y)
    return (a, b)

def combine(a, b):
    return list(zip(a, b))

print(split([(1, 2), (3, 4), (5, 6)]))
print(combine([1, 3, 5], [2, 4, 6]))

def partition(f, lst):
    yes = []
    no = []
    for x in lst:
        if f(x):
            yes.append(x)
        else:
            no.append(x)
    return (yes, no)

print(partition(lambda x: x % 2 == 0, [1,2,3,4,5,6,7,8,9]))
def liste_cifre(lista):
    res = []
    for n in lista:
        s = []
        for ch in str(n):
            if ch.isdigit():
                d = int(ch)
                if d % 2 == 0:
                    s.append(d)
        res.append(s)
    return res
print(liste_cifre([123, 456, 789, 2468]))    
 
    '''



        


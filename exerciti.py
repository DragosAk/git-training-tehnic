lista_smen = [4,2, 8, 6 ,7]
lista_mai_smen = list(map(lambda x: x + 3, lista_smen))
print(lista_mai_smen)
lista_mai_smen.sort()
print(lista_mai_smen)
import functools, operator
suma = functools.reduce(lambda x, y: x + y, lista_mai_smen)
print(suma)
#lista_filtrata = list(filter(lambda x: x % 2 == 0, lista_mai_smen))
lista_mai_smen.sort(reverse = True)
print(lista_mai_smen)
'''

'''#part1
#4,5,6,7
#4
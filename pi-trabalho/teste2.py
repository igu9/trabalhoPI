import sys

def not_main():
    print("######not_main######")
    

def main():
    print("%%%%%%%main%%%%%%%%%%%")
    print("argumentos recebidos:")
    for x in sys.argv:
        print(x)

    print(len(sys.argv))

if __name__ == '__main__':
    main()
else:
    not_main()
SIZE = 515
f = open(f'BB-{SIZE}/D_original.txt', "r")


lines = f.read().split('\n')
f.close()
f = open('D.txt', 'w')
for i in range(0, SIZE):
    row = []
    for j in range(0, SIZE):
        row.append(lines[(i * SIZE) + j])

    f.write(';'.join(row) + '\n')
f.close()
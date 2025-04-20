def create_minefield(N):
    minefield = [[0 for _ in range(N)] for _ in range(N)]
    return minefield

def place_mines(minefield, N):
    print(f"{N}x{N} 크기의 지뢰 필드를 생성합니다. 지뢰를 배치하세요 (1: 지뢰, 0: 빈 칸):")
    for i in range(N):
        row = input().strip()
        if len(row) != N:
            print(f"입력된 행의 길이가 {N}이 아닙니다. 다시 입력하세요.")
            return place_mines(minefield, N)
        minefield[i] = [int(char) for char in row]

def count_adjacent_mines(minefield, N, x, y):
    count = 0
    for i in range(max(0, x-1), min(N, x+2)):
        for j in range(max(0, y-1), min(N, y+2)):
            if (i != x or j != y) and minefield[i][j] == 1:
                count += 1
    return count

def print_minefield(minefield, N):
    for i in range(N):
        for j in range(N):
            if minefield[i][j] == 1:
                print("*", end=" ")
            else:
                print(count_adjacent_mines(minefield, N, i, j), end=" ")
        print()

N = int(input("지뢰 필드의 크기 N을 입력하세요: "))
minefield = create_minefield(N)
place_mines(minefield, N)
print("결과:")
print_minefield(minefield, N)
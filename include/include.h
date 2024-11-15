#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/Reverse.h>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>
using namespace std;
using namespace Eigen;

using Actions          = vector<pair<int, int>>;
// 定义状态类型
enum StateType { NORMAL, FORBIDDEN, TARGET };

struct config {
    int forbidden;
    int bound;
    int target;
    int other;
};
// 使用宏定义方向坐标
#define LEFT                                                                                       \
    { 0, -1 }
#define RIGHT                                                                                      \
    { 0, 1 }
#define UP                                                                                         \
    { -1, 0 }
#define DOWN                                                                                       \
    { 1, 0 }
#define STAY                                                                                       \
    { 0, 0 }

// 获取状态索引
inline int getStateIndex(int x, int y, int gridWidth) {
    return x * gridWidth + y;
}
inline string getArrow(int i_row, int i_col, int j_row, int j_col) {
    // Determine the direction of the arrow based on row and column differences
    int row_diff = j_row - i_row;
    int col_diff = j_col - i_col;

    // Vertical and horizontal movement
    if (row_diff == 0 && col_diff > 0) return "→"; // Right
    if (row_diff == 0 && col_diff < 0) return "←"; // Left
    if (row_diff > 0 && col_diff == 0) return "↓"; // Down
    if (row_diff < 0 && col_diff == 0) return "↑"; // Up

    return " "; // No movement (if they are the same)
}

inline void drawMatrixMovement(const MatrixXd& P) {
    const int N = 5; // 5x5 matrix size
    vector<vector<string>> matrix(N, vector<string>(N, " ")); // 5x5 grid for display

    // Iterate over the 25x25 matrix P
    for (int i = 0; i < N * N; ++i) {
        for (int j = 0; j < N * N; ++j) {
            if (P(i, j) == 1) {
                int i_row = i / N, i_col = i % N; // Convert i to 5x5 indices
                int j_row = j / N, j_col = j % N; // Convert j to 5x5 indices

                if (i_row == j_row && i_col == j_col) {
                    matrix[i_row][i_col] = "o"; // No movement
                } else {
                    // Get the appropriate arrow for the movement direction
                    matrix[i_row][i_col] = getArrow(i_row, i_col, j_row, j_col);
                }
            }
        }
    }

    // Print the movement map
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << setw(5) << matrix[i][j];
        }
        cout << endl;
    }
}
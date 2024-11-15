#include "include.h"
#include <Eigen/src/Core/Matrix.h>

static const int    gridSize  = 5;                   // 网格大小 (5x5)
static const int    numStates = gridSize * gridSize; // 总状态数
static const double _gamma    = 0.9;                 // 折扣因子
static const double tolerance = 1e-3;

static vector<vector<StateType>> grid = {{NORMAL, NORMAL, NORMAL, NORMAL, NORMAL},
                                         {NORMAL, FORBIDDEN, FORBIDDEN, NORMAL, NORMAL},
                                         {NORMAL, NORMAL, FORBIDDEN, NORMAL, NORMAL},
                                         {NORMAL, FORBIDDEN, TARGET, FORBIDDEN, NORMAL},
                                         {NORMAL, FORBIDDEN, NORMAL, NORMAL, NORMAL}};

// 初始策略
static vector<vector<pair<int, int>>> Policy = {{STAY, STAY, STAY, STAY, STAY},
                                                {STAY, STAY, STAY, STAY, STAY},
                                                {STAY, STAY, STAY, STAY, STAY},
                                                {STAY, STAY, STAY, STAY, STAY},
                                                {STAY, STAY, STAY, STAY, STAY}};
static config                         award{-10, -1, 1, 0};

static vector<pair<int, int>> acts{LEFT, UP, RIGHT, DOWN, STAY};

extern VectorXd valueIteration(Actions acts, const vector<vector<StateType>> &grid, double GAMMA,
                               int NUM_STATES, config config = {-1, -1, 1, 0},
                               int maxIterations = 2000, double tolerance = 1e-6);

// 迭代求解贝尔曼方程
inline VectorXd iterativeSolution(const MatrixXd &P, const VectorXd &r, VectorXd v,int maxIterations = 1000,
                                  double tolerance = 1e-6) {
    // VectorXd v = VectorXd::Zero(numStates);
    for (int k = 0; k < maxIterations; ++k) {
        VectorXd vNext = r + _gamma * P * v;
        if ((vNext - v).norm() < tolerance) {
            // cout << "value iteration converged after " << k + 1 << " iterations." << endl;
            return vNext;
        }
        v = vNext;
    }
    // cout << "Reached maximum iterations without convergence." << endl;
    return v;
}

static VectorXd strategyIteration(Actions acts, const vector<vector<StateType>> &grid, double GAMMA,
                                  int NUM_STATES, config config = {-1, -1, 1, 0},
                                  int truncate_at = 2000, int maxIterations = 2000,
                                  double tolerance = 1e-6) {

    VectorXd r(NUM_STATES); // immediate return vector
    VectorXd v(NUM_STATES); // state value vector
    // 初始时全部为stay 转移矩阵为单位矩阵
    MatrixXd P = MatrixXd::Identity(NUM_STATES, NUM_STATES);
    r.setZero();
    MatrixXd QSA(NUM_STATES, acts.size()); // QSA(i,j)表示状态i采取动作acts(j)得到的回报
    for (int k = 0; k < maxIterations; k++) {
        v = iterativeSolution(P, r, v,truncate_at);
        for (int i = 0; i < NUM_STATES; i++) { // 得到最大的策略
            for (int j = 0; j < acts.size(); j++) {
                auto [offsetx, offsety] = acts[j];
                int x                   = i / grid[0].size();
                int y                   = i % grid[0].size();
                int newX                = x + offsetx;
                int newY                = y + offsety;
                int immediate_value     = config.other;
                // 设置奖励
                // 超过边界
                if (newX < 0 || newX >= grid.size() || newY < 0 || newY >= grid[0].size()) {
                    immediate_value = config.bound;
                    newX            = x; // 停在原址
                    newY            = y;
                } else if (grid[newX][newY] == TARGET) {
                    immediate_value = config.target;
                } else if (grid[newX][newY] == FORBIDDEN) {
                    immediate_value = config.forbidden;
                }
                // immediate return + gamma*v(s)
                QSA(i, j) = immediate_value + GAMMA * v(getStateIndex(newX, newY, grid[0].size()));
            }
        }
        // 更新矩阵 P
        int m = QSA.rows(); // 状态数量
        int n = QSA.cols(); // 操作数量

        for (int i = 0; i < m; ++i) {
            double max_score   = QSA(i, 0);
            int    best_action = 0;

            // 找到最大得分对应的操作
            for (int j = 1; j < n; ++j) {
                if (QSA(i, j) > max_score) {
                    max_score   = QSA(i, j);
                    best_action = j;
                }
            }

            P.row(i).setZero(); // 先将当前行的概率清零
            auto [offsetx, offsety] = acts[best_action];
            int x                   = i / grid[0].size();
            int y                   = i % grid[0].size();
            int newX                = x + offsetx;
            int newY                = y + offsety;
            P(i, newX * grid[0].size() + newY) = 1.0; // 设置从状态 i 转移到 new_state 的概率为 1
        }
        // 更新r
        for (int idx = 0; idx < NUM_STATES; idx++) {
            // P(r,?)=1
            for (int t = 0; t < NUM_STATES; t++) {
                if (P(idx, t) == 1.0) {
                    int newX = t / grid[0].size();
                    int newY = t % grid[0].size();
                    if (newX < 0 || newX >= grid.size() || newY < 0 || newY >= grid[0].size()) {
                        r(idx) = config.bound;
                    } else if (grid[newX][newY] == TARGET) {
                        r(idx) = config.target;
                    } else if (grid[newX][newY] == FORBIDDEN) {
                        r(idx) = config.forbidden;
                    } else {
                        r(idx) = config.other;
                    }
                }
            }
        }
        VectorXd vNext = iterativeSolution(P, r,v,truncate_at);
        if ((vNext - v).norm() < tolerance) {
            cout << "Converged after " << k + 1 << " iterations." << endl;
            drawMatrixMovement(P);
            return vNext;
        }
        v = vNext;
    }
    cout << "Reached maximum iterations without convergence." << endl;
    return v;

    // return;
}
void test_valueIteration() {
    cout << "value iteration " << endl << endl;
    auto            ans = valueIteration(acts, grid, 0.9, numStates, {-10, -1, 1, 0}, 1000, 1e-3);
    Eigen::MatrixXd m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
    cout << m.transpose() << endl << endl;
}

void test_policyIteration() {
    cout << "policy iteration: " << endl << endl;
    auto            ans = strategyIteration(acts, grid, 0.9, numStates, award);
    Eigen::MatrixXd m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
    cout << m.transpose() << endl << endl;
}

void test_truncatedIteration(int iteration_times) {
    cout << "truncated policy iteration: " << endl << endl;
    auto            ans = strategyIteration(acts, grid, 0.9, numStates, award, iteration_times);
    Eigen::MatrixXd m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
    cout << m.transpose() << endl << endl;
}

int hw2_test() {

    test_valueIteration();
    test_policyIteration();
    test_truncatedIteration(5);
    return 0;
}

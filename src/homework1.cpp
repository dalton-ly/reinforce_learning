#include "include.h"
#include <vector>

static const int    gridSize  = 5;                   // 网格大小 (5x5)
static const int    numStates = gridSize * gridSize; // 总状态数
static const double _gamma    = 0.9;                 // 折扣因子

vector<vector<pair<int, int>>> getStrategyDirections(int strategy) {
    vector<vector<pair<int, int>>> directions(gridSize, vector<pair<int, int>>(gridSize, STAY));

    switch (strategy) {
    case 1:
        directions = {{RIGHT, RIGHT, RIGHT, DOWN, DOWN},
                      {UP, UP, RIGHT, DOWN, DOWN},
                      {UP, LEFT, DOWN, RIGHT, DOWN},
                      {UP, RIGHT, STAY, LEFT, DOWN},
                      {UP, RIGHT, UP, LEFT, LEFT}};
        break;
    case 2:
        directions = {{RIGHT, RIGHT, RIGHT, RIGHT, DOWN},
                      {UP, UP, RIGHT, RIGHT, DOWN},
                      {UP, LEFT, DOWN, RIGHT, DOWN},
                      {UP, RIGHT, STAY, LEFT, DOWN},
                      {UP, RIGHT, UP, LEFT, LEFT}};
        break;
    case 3:
        directions = {{RIGHT, RIGHT, RIGHT, RIGHT, RIGHT},
                      {RIGHT, RIGHT, RIGHT, RIGHT, RIGHT},
                      {RIGHT, RIGHT, RIGHT, RIGHT, RIGHT},
                      {RIGHT, RIGHT, RIGHT, RIGHT, RIGHT},
                      {RIGHT, RIGHT, RIGHT, RIGHT, RIGHT}};
        break;
    case 4:
        directions = {{RIGHT, LEFT, LEFT, UP, UP},
                      {DOWN, STAY, RIGHT, DOWN, RIGHT},
                      {LEFT, RIGHT, DOWN, LEFT, STAY},
                      {STAY, DOWN, UP, UP, RIGHT},
                      {STAY, RIGHT, STAY, RIGHT, STAY}};
        break;
    }
    return directions;
}
// 初始化奖励向量 r_x 和转移概率矩阵 P_x
void initializeGrid(MatrixXd &P, VectorXd &r, const vector<vector<StateType>> &grid,
                    const vector<vector<pair<int, int>>> &directions) {
    r.setZero();
    P.setZero();

    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            int idx     = getStateIndex(row, col, grid.size());
            auto [x, y] = directions[row][col];
            int newX    = row + x;
            int newY    = col + y;

            // 设置奖励
            // 超过边界
            if (newX < 0 || newX >= grid.size() || newY < 0 || newY >= grid.size()) {
                r(idx) = -1;
            } else if (grid[newX][newY] == TARGET) {
                r(idx) = 1.0;
            } else if (grid[newX][newY] == FORBIDDEN) {
                r(idx) = -1.0;
            } else {
                r(idx) = 0.0;
            }
            // 设置转移概率

            int origin_index     = getStateIndex(row, col, grid.size());
            int after_transition = getStateIndex(newX, newY, grid.size());

            if (newX < 0 || newX >= grid.size() || newY < 0 || newY >= grid.size()) {
                P(origin_index, origin_index) = 1; // 越界
                continue;
            }
            P(origin_index, after_transition) = 1; // 转移到位置的概率
        }
    }
}

// 封闭解法
VectorXd closedFormSolution(const MatrixXd &P, const VectorXd &r) {
    MatrixXd I = MatrixXd::Identity(numStates, numStates);
    return (I - _gamma * P).inverse() * r;
}

// 迭代解法
static VectorXd iterativeSolution(const MatrixXd &P, const VectorXd &r, int maxIterations = 1000,
                                  double tolerance = 1e-3) {
    VectorXd v = VectorXd::Zero(numStates);
    for (int k = 0; k < maxIterations; ++k) {
        VectorXd vNext = r + _gamma * P * v;
        if ((vNext - v).norm() < tolerance) {
            cout << "Converged after " << k + 1 << " iterations." << endl;
            return vNext;
        }
        v = vNext;
    }
    cout << "Reached maximum iterations without convergence." << endl;
    return v;
}

pair<VectorXd, vector<double>> valueIteration(Actions acts, const vector<vector<StateType>> &grid,
                                              double GAMMA, int NUM_STATES,
                                              config config     = {-1, -1, 1, 0},
                                              int maxIterations = 2000, double tolerance = 1e-3) {

    vector<double> statevalue;
    VectorXd       r(NUM_STATES); // immediate return vector
    VectorXd       v(NUM_STATES);
    MatrixXd       P(NUM_STATES, NUM_STATES);
    v.setZero();
    r.setZero();
    MatrixXd QSA(NUM_STATES, acts.size());
    for (int k = 0; k < maxIterations; k++) {
        statevalue.emplace_back(v(4));
        for (int i = 0; i < NUM_STATES; i++) {
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
        VectorXd vNext = r + GAMMA * P * v;
        if ((vNext - v).norm() < tolerance) {
            cout << "Converged after " << k + 1 << " iterations." << endl;
            drawMatrixMovement(P);
            return {vNext, statevalue};
        }
        v = vNext;
    }
    cout << "Reached maximum iterations without convergence." << endl;
    return {v, statevalue};
}

int hw1_test() {
    const int    gridSize  = 5;                   // 网格大小 (5x5)
    const int    numStates = gridSize * gridSize; // 总状态数
    const double _gamma    = 0.9;                 // 折扣因子
    using Actions          = vector<pair<int, int>>;
    // 定义网格状态
    vector<vector<StateType>> grid = {{NORMAL, NORMAL, NORMAL, NORMAL, NORMAL},
                                      {NORMAL, FORBIDDEN, FORBIDDEN, NORMAL, NORMAL},
                                      {NORMAL, NORMAL, FORBIDDEN, NORMAL, NORMAL},
                                      {NORMAL, FORBIDDEN, TARGET, FORBIDDEN, NORMAL},
                                      {NORMAL, FORBIDDEN, NORMAL, NORMAL, NORMAL}};

    vector<vector<StateType>> grid3_1 = {{NORMAL, TARGET, NORMAL}};

    MatrixXd P(numStates, numStates);
    VectorXd r(numStates);
    // Homework1
    for (int strategy = 1; strategy <= 4; strategy++) {
        auto directions = getStrategyDirections(strategy);
        initializeGrid(P, r, grid, directions);
        cout << "Using Strategy " << strategy << endl;
        auto            ans = iterativeSolution(P, r);
        Eigen::MatrixXd m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
        cout << m.transpose() << endl;
        cout << endl;
    }

    vector<pair<int, int>> acts{LEFT, UP, RIGHT, DOWN, STAY};
    vector<pair<int, int>> acts3_1{LEFT, STAY, RIGHT};
    cout << "using value iteration algorithm to solve bellman optimality equation" << endl;
    // 2.1 3*1 world
    cout << "solve in : 1x3 Grid world, gamma: 0.9" << endl << endl;
    auto [ans1_3, _] = valueIteration(acts3_1, grid3_1, 0.9, 3);
    cout << ans1_3.transpose() << endl << endl << endl;

    cout << "solve in : 5x5 Grid world, gamma: 0.9" << endl << endl;
    auto            ans = valueIteration(acts, grid, 0.9, numStates).first;
    Eigen::MatrixXd m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
    cout << m.transpose() << endl << endl;

    cout << "solve in : 5x5 Grid world, gamma: 0.5" << endl << endl;
    ans = valueIteration(acts, grid, 0.5, numStates).first;
    m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
    cout << m.transpose() << endl << endl;

    cout << "solve in : 5x5 Grid world, gamma: 0" << endl << endl;
    ans = valueIteration(acts, grid, 0, numStates).first;
    m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
    cout << m.transpose() << endl << endl;

    cout << "solve in : 5x5 Grid world, gamma: 0.9,  rforbidden = -10" << endl << endl;
    ans = valueIteration(acts, grid, 0.9, numStates, config{-10, -1, 1, 0}).first;
    m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
    cout << m.transpose() << endl << endl;

    cout << "solve in : 5x5 Grid world, gamma: 0.9,   rboundary = rforbidden = 0, rtarget = 2, "
            "rotherstep = 1"
         << endl
         << endl;
    ans = valueIteration(acts, grid, 0.9, numStates, config{0, 0, 2, 1}).first;
    m   = Eigen::Map<Eigen::MatrixXd>(ans.data(), 5, 5);
    cout << m.transpose() << endl << endl;

    return 0;
}

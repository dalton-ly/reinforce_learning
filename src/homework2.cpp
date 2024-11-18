#include "include.h"
#include "matplotlibcpp.h"
#include <Eigen/src/Core/Matrix.h>
#include <utility>
#include <vector>

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

extern pair<VectorXd, vector<double>>
valueIteration(Actions acts, const vector<vector<StateType>> &grid, double GAMMA, int NUM_STATES,
               config config = {-1, -1, 1, 0}, int maxIterations = 2000, double tolerance = 1e-6);

// 迭代求解贝尔曼方程
inline VectorXd iterativeSolution(const MatrixXd &P, const VectorXd &r, VectorXd v,
                                  int maxIterations = 1000, double tolerance = 1e-6) {
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

static pair<VectorXd, vector<double>>
strategyIteration(Actions acts, const vector<vector<StateType>> &grid, double GAMMA, int NUM_STATES,
                  config config = {-1, -1, 1, 0}, int truncate_at = 2000, int maxIterations = 2000,
                  double tolerance = 1e-3) {

    VectorXd       r(NUM_STATES); // immediate return vector
    VectorXd       v(NUM_STATES); // state value vector
    vector<double> statevalue;
    // 初始时全部为stay 转移矩阵为单位矩阵
    MatrixXd P = MatrixXd::Identity(NUM_STATES, NUM_STATES);
    r.setZero();
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
    v = iterativeSolution(P, r, v);
    // statevalue.emplace_back(0);
    MatrixXd QSA(NUM_STATES, acts.size()); // QSA(i,j)表示状态i采取动作acts(j)得到的回报
    for (int k = 0; k < maxIterations; k++) {
        statevalue.emplace_back(v(4));
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
            // cout<<"i: "<<i<<" newX: "<<newX<<" grid[0],size() "<< grid[0].size() <<" newY "<<newY<<endl;
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
        VectorXd vNext = iterativeSolution(P, r, v, truncate_at);
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

void plot_vectors(const std::vector<double> &v1, const std::vector<double> &v2,
                  const std::vector<double> &v3) {
    // 获取最大的长度，确保所有的vector都能匹配
    size_t max_len = std::max({v1.size(), v2.size(), v3.size()});

    // 填充至20个点
    const size_t num_points = 30;

    std::vector<double> x(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        x[i] = i;
    }

    auto fill_or_truncate = [max_len](const std::vector<double> &v) {
        std::vector<double> result(num_points, 0.0);
        size_t              len = v.size();
        for (size_t i = 0; i < num_points; ++i) {
            if (i < len) {
                result[i] = v[i];
            } else {
                result[i] = v[len - 1]; // 填充最后一个值
            }
        }
        return result;
    };

    // 填充数据
    auto v1_filled = fill_or_truncate(v1);
    auto v2_filled = fill_or_truncate(v2);
    auto v3_filled = fill_or_truncate(v3);

    // 绘制图形
    plt::plot(x, v1_filled, {{"label", "value iteration"}});
    plt::plot(x, v2_filled, {{"label", "policy iteration"}});
    plt::plot(x, v3_filled, {{"label", "truncated policy iteration"}});

    // 设置标题和标签
    plt::title("Plot of Vectors");
    plt::xlabel("Index");
    plt::ylabel("Value");

    // 添加图例
    plt::legend();
    // plt::save("output.png"); // 保存图像到文件
    // // 显示图形
    plt::show();
}

void compare_methods() {
    cout << "value iteration " << endl << endl;
    auto [ans_value, s_value] = valueIteration(acts, grid, 0.9, numStates, award);
    Eigen::MatrixXd m         = Eigen::Map<Eigen::MatrixXd>(ans_value.data(), 5, 5);
    cout << m.transpose() << endl << endl;

    cout << "policy iteration: " << endl << endl;
    auto [ans_policy, s_policy] = strategyIteration(acts, grid, 0.9, numStates, award);
    m                           = Eigen::Map<Eigen::MatrixXd>(ans_policy.data(), 5, 5);
    cout << m.transpose() << endl << endl;

    int iteration_times = 2;
    cout << "truncated policy iteration: " << endl << endl;
    auto [ans_truncated, s_truncated] =
        strategyIteration(acts, grid, 0.9, numStates, award, iteration_times);
    m = Eigen::Map<Eigen::MatrixXd>(ans_truncated.data(), 5, 5);
    cout << m.transpose() << endl << endl;

    plot_vectors(s_value, s_policy, s_truncated);
}

void compare_iter_times() {
    vector<int>            times{1, 5, 9, 56};
    const double           optimal = 5.31441;
    vector<vector<double>> statevalues;
    for (int iteration_times : times) {
        cout << "truncated policy iteration, truncate at " << iteration_times << " : " << endl
             << endl;
        auto [ans_truncated, s_truncated] =
            strategyIteration(acts, grid, 0.9, numStates, award, iteration_times);
        auto m = Eigen::Map<Eigen::MatrixXd>(ans_truncated.data(), 5, 5);
        cout << m.transpose() << endl << endl;
        statevalues.emplace_back(s_truncated);
    }
    // 获取最大的长度，确保所有的vector都能匹配
    size_t max_len = 0;
    for (vector<double> &vec : statevalues) {
        max_len = max(max_len, vec.size());
    }

    // 填充至20个点
    const size_t num_points = 10;

    std::vector<double> x(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        x[i] = i;
    }
    for (size_t i = 0; i < statevalues.size(); ++i) {
        const auto         &vec = statevalues[i];
        std::vector<double> diff(num_points, 0.0); // 初始化差值数组

        for (size_t j = 0; j < num_points; ++j) {
            if (j < vec.size()) {
                diff[j] = vec[j] - optimal; // 计算差值
            } else if (!vec.empty()) {
                diff[j] = vec.back() - optimal; // 填充并计算差值
            }
        }

        // 绘制差值曲线，每条曲线都有唯一的标签
        plt::plot(x, diff, {{"label", "iteration times= " + std::to_string(times[i])}});
    }

    // 设置标题和标签
    plt::title("Plot of Vectors");
    plt::xlabel("Index");
    plt::ylabel("Value");

    // 添加图例
    plt::legend();

    // 显示图形
    plt::show();
}

int hw2_test() {
    compare_methods();
    compare_iter_times();
    return 0;
}

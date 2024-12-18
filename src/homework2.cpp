#include "include.h"
#include "tqdm.h"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace fs                  = std::filesystem;
static const int    gridSize  = 5;                   // 网格大小 (5x5)
static const int    numStates = gridSize * gridSize; // 总状态数
static const double _gamma    = 0.9;                 // 折扣因子
static const double tolerance = 1e-3;
mt19937             rng(static_cast<unsigned int>(time(nullptr)));

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
static VectorXd iterativeSolution(const MatrixXd &P, const VectorXd &r, VectorXd v,
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
            // cout<<"i: "<<i<<" newX: "<<newX<<" grid[0],size() "<< grid[0].size() <<" newY
            // "<<newY<<endl;
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

// ε-贪心策略选择动作 最大的概率选择q-value最大对应的动作
int epsilon_greedy(const vector<double> &q_values, double EPSILON, int ACTIONS) {
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto   max_action = distance(q_values.begin(), max_element(q_values.begin(), q_values.end()));
    double minprob    = double(4.0 / 5.0) * EPSILON;
    if (dist(rng) < minprob) {
        // 探索：随机选择动作
        uniform_int_distribution<int> action_dist(0, ACTIONS - 1);
        auto                          tmp = action_dist(rng);
        while (tmp == max_action) {
            tmp = action_dist(rng);
        }
        return tmp;
    }
    // 利用：选择最优动作
    return max_action;
}

tuple<int, int, double> step(Actions acts, const vector<vector<StateType>> &grid, int x, int y,
                             int action, config reward = {-10, -1, 1, 0}) {
    int nx = x + acts[action].first;
    int ny = y + acts[action].second;
    if (nx < 0 || nx >= gridSize || ny < 0 || ny >= gridSize) {
        return {x, y, reward.bound};
    }
    if (grid[nx][ny] == FORBIDDEN) {
        return {nx, ny, reward.forbidden};
    }
    if (grid[nx][ny] == TARGET) {
        return {nx, ny, reward.target};
    }
    return {nx, ny, reward.other};
}

MatrixXd MC_Greedy(Actions acts, const vector<vector<StateType>> &grid,
                   const vector<vector<int>> &Policy, double GAMMA, double EPSILON,
                   const int MAX_STEPS = 1e6, const int EPISODES = 100,
                   config config = {-10, -1, 1, 0}) {

    vector<vector<vector<double>>> Q(
        gridSize, vector<vector<double>>(gridSize, vector<double>(acts.size(), 0)));
    // Q(x,y)保存坐标为x,y的对应的每个动作

    // 开始生成episode
    tqdm                        bar;
    vector<vector<vector<int>>> visit_count(5, vector<vector<int>>(5, vector<int>(5, 0)));
    for (int episode = 0; episode < EPISODES; episode++) {
        bar.progress(episode, EPISODES);
        // cout << " episode: " << episode << endl;
        // 记录轨迹
        vector<tuple<int, int, int, double>> episode_data; // 在(x,y)处选择的action以及对应的回报
        int x = rng() % gridSize;
        int y = rng() % gridSize;
        // int a = rng() % 5;
        // int x=0,y=0;
        int steps = 0;
        while (steps < MAX_STEPS ) {
            // if (steps % 10000 == 0) {
            //     cout << " steps: " << steps << endl;
            // }
            int    action = (episode == 0 ? Policy[x][y] : epsilon_greedy(Q[x][y], EPSILON, 5));
            int    nx, ny;
            double reward;
            tie(nx, ny, reward) = step(acts, grid, x, y, action);
            // 储存
            episode_data.emplace_back(x, y, action, reward);
            x = nx;
            y = ny;
            steps++;
        }

        // 累计回报
        double G = 0.0;
        std::reverse(episode_data.begin(), episode_data.end());
        // cout << "begin update " << endl;
        // int cnt = 0;
        for (const auto &[sx, sy, a, r] : episode_data) {
            // cnt++;
            // if (cnt % 10000 == 0) {
            //     cout << " cnts: " << cnt << endl;
            // }
            G = r + GAMMA * G;
            // 更新访问次数
            visit_count[sx][sy][a] += 1;
            int count = visit_count[sx][sy][a];

            // 增量更新 Q 值
            Q[sx][sy][a] += (G - Q[sx][sy][a]) /(double) count;
        }
    }
    // 保存为 CSV 文件
    fs::path current_path = fs::current_path();         // 当前工作目录
    fs::path parent_path  = current_path.parent_path(); // 上级目录

    // 构建文件路径
    string outputfilename = (parent_path / "data" / to_string(EPSILON) /
                             ("visit_count_" + to_string(MAX_STEPS) + ".csv"))
                                .string();
    ofstream outfile(outputfilename);
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            for (int a = 0; a < 5; ++a) {
                outfile << visit_count[x][y][a];
                if (a < 4)
                    outfile << ","; // 不同动作用逗号分隔
            }
            outfile << "\n"; // 每行对应一个 (x, y) 点
        }
    }
    outfile.close();

    // 根据 ε-贪心策略计算状态值矩阵
    MatrixXd statevalue(5, 5);
    statevalue.setZero();

    string best_action_name =
        (parent_path / "data" / ("best_action_of" + to_string(EPSILON) + ".csv")).string();
    string state_val =
        (parent_path / "data" / ("state_value_of" + to_string(EPSILON) + ".csv")).string();

    ofstream out(best_action_name);
    ofstream outstate_val(state_val);

    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            // 找到 Q(x, y, a) 中的最优动作的索引
            int best_action =
                distance(Q[x][y].begin(), max_element(Q[x][y].begin(), Q[x][y].end()));
            out << best_action;
            // 计算状态值 V(s) = ∑ π(a|s) Q(s, a)
            double state_value = 0.0;
            for (int a = 0; a < 5; ++a) {
                double pi_a_s = (a == best_action) ? (1 - EPSILON + EPSILON / 5) : (EPSILON / 5);
                state_value += pi_a_s * Q[x][y][a];
            }
            statevalue(x, y) = state_value;
            outstate_val << state_value;
            if (y < 4) {
                out << ",";
                outstate_val << ",";
            }
        }
        out << "\n";
        outstate_val << "\n";
    }
    out.close();
    outstate_val.close();

    return statevalue;
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

    // 保存 s_value 到文件
    std::ofstream value_file("s_value.txt");
    for (auto v : s_value) {
        value_file << v << " ";
    }
    value_file.close();

    // 保存 s_policy 到文件
    std::ofstream policy_file("s_policy.txt");
    for (auto v : s_policy) {
        policy_file << v << " ";
    }
    policy_file.close();

    // 保存 s_truncated 到文件
    std::ofstream truncated_file("s_truncated.txt");
    for (auto v : s_truncated) {
        truncated_file << v << " ";
    }
    truncated_file.close();
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

    // 保存 statevalues 到文件
    std::ofstream statevalues_file("statevalues.txt");
    for (const auto &s_truncated : statevalues) {
        for (auto v : s_truncated) {
            statevalues_file << v << " ";
        }
        statevalues_file << "\n";
    }
    statevalues_file.close();
}

void test_MC_greedy() {
    vector<vector<int>> policy(5, vector<int>(5, 0));
    int                 left = 0, up = 1, right = 2, down = 3, stay = 4;
    policy = {{left, left, left, left, down},
              {up, up, right, right, down},
              {up, left, down, right, down},
              {up, right, stay, left, down},
              {up, right, up, left, left}};
    // double epsilon = 0.5;
    // for (int steps : vector<int>{100, 1000, 10000, 1000000}) {
    //     auto result = MC_Greedy(acts, grid, policy, _gamma, epsilon, steps, 3);
    // }
    // cout << result << endl;
    for (double epsilon : vector<double>{0.0, 0.1, 0.2, 0.5}) {
        MC_Greedy(acts, grid, policy, _gamma, epsilon, 1e5, 10000);
    }
}

int hw2_test() {
    // compare_methods();
    // compare_iter_times();
    test_MC_greedy();
    return 0;
}

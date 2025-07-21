// overhead.cpp
// -----------------------------------------------------------------------------
// Full control-loop timing harness with N interacting tasks.
//   u = −K·e               (K is N×N gain matrix)
//   P_i ← P_i /(1+P_i Σ_{j≠i} P_j |u_i|)  (cross-coupled period law)
// All overheads reported in µs. Tested under g++-9 (C++17).
// -----------------------------------------------------------------------------
// Build:  g++ -std=c++17 -O2 -pthread overhead.cpp -o overhead
// -----------------------------------------------------------------------------

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <deque>
#include <numeric>
#include <algorithm>
#include <thread>
#include <random>
#include <string>
#include <atomic>
#include <cmath>
#include <iomanip>

using namespace std;
using Clock        = chrono::high_resolution_clock;
using Microseconds = chrono::microseconds;
using namespace std::string_literals;

struct SharedData {
    vector<float> values;
    vector<float> executiontime;
    vector<float> newperiods; // ms
};

// --------------------------- median filter helpers ---------------------------
static void bubbleSort(vector<float>& v) {
    for (size_t i = 0; i + 1 < v.size(); ++i)
        for (size_t j = 0; j + 1 < v.size() - i; ++j)
            if (v[j] > v[j + 1]) swap(v[j], v[j + 1]);
}

static double medianFilter(double x, deque<float>& w, size_t ws) {
    w.push_back(x);
    if (w.size() > ws) w.pop_front();
    vector<float> t(w.begin(), w.end());
    bubbleSort(t);
    return t[t.size() / 2];
}

static void applyMedianFilter(vector<float>& v, size_t ws) {
    deque<float> w;
    for (float& x : v)
        x = medianFilter(x, w, ws);
}

// --------------------------- stats helper (µs) -------------------------------
static pair<double, double> stats(const vector<double>& v) {
    if (v.empty()) return {0.0, 0.0};
    double m = accumulate(v.begin(), v.end(), 0.0) / v.size();
    double sq = inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double var = sq / v.size() - m * m;
    return {m, sqrt(max(0.0, var))};
}

int main() {
    const vector<int> TASK_COUNTS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const int CONTROL_PERIODS = 100;
    const size_t WINDOW_SIZE = 8;
    const double PERIOD_DURATION = 0.1;
    const float SETPOINT = 50.0f;
    const float BASE_GAIN = 0.05f;

    mt19937 rng(random_device{}());
    uniform_real_distribution<float> distVal(0.0f, 100.0f);
    uniform_real_distribution<float> distTime(0.0f, 10.0f);

    cout << left << setw(8) << "Tasks"
         << setw(15) << "Meas(us)"
         << setw(15) << "Meas"
         << setw(15) << "Ctrl(us)"
         << setw(15) << "Ctrl"
         << setw(15) << "Act(us)"
         << setw(15) << "Act" << '\n'
         << string(8 + 15 * 6, '-') << '\n';

    for (int tasks : TASK_COUNTS) {
        SharedData sd;
        sd.values.assign(tasks, 0.0f);
        sd.executiontime.assign(tasks, 0.0f);
        sd.newperiods.assign(tasks, 1.0f);

        vector<vector<float>> K(tasks, vector<float>(tasks, 0.0f));
        for (int i = 0; i < tasks; ++i) K[i][i] = BASE_GAIN;

        atomic<bool> done(false);
        thread producer([&]() {
            while (!done.load()) {
                for (int i = 0; i < tasks; ++i) {
                    sd.values[i] = distVal(rng);
                    sd.executiontime[i] = distTime(rng);
                }
                this_thread::sleep_for(chrono::milliseconds(100));
            }
        });

        vector<double> measOH, ctrlOH, actOH;
        measOH.reserve(CONTROL_PERIODS);
        ctrlOH.reserve(CONTROL_PERIODS);
        actOH.reserve(CONTROL_PERIODS);

        for (int p = 0; p < CONTROL_PERIODS; ++p) {
            // Measurement phase
            vector<vector<float>> execCnt(tasks);
            vector<double> measDur;
            measDur.reserve(static_cast<size_t>(PERIOD_DURATION * 10));

            auto tend = Clock::now() + chrono::duration<double>(PERIOD_DURATION);
            while (Clock::now() < tend) {
                auto t0 = Clock::now();
                for (int i = 0; i < tasks; ++i) {
                    float v = sd.values[i];
                    execCnt[i].push_back(v / (sd.newperiods[i] * 1000.0f + 1e-6f));
                }
                auto t1 = Clock::now();
                measDur.push_back(chrono::duration_cast<Microseconds>(t1 - t0).count());
                this_thread::sleep_for(chrono::microseconds(1000));
            }
            measOH.push_back(stats(measDur).first);
            for (int i = 0; i < tasks; ++i) applyMedianFilter(execCnt[i], WINDOW_SIZE);

            // Controller
            auto c0 = Clock::now();
            vector<float> error(tasks), u(tasks);
            for (int i = 0; i < tasks; ++i) {
                float avg = 0.0f;
                if (!execCnt[i].empty())
                    avg = accumulate(execCnt[i].begin(), execCnt[i].end(), 0.0f) / execCnt[i].size();
                error[i] = SETPOINT - avg;
            }
            for (int i = 0; i < tasks; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < tasks; ++j)
                    acc += K[i][j] * error[j];
                u[i] = -acc;
            }
            auto c1 = Clock::now();
            ctrlOH.push_back(chrono::duration_cast<Microseconds>(c1 - c0).count());

            // Actuation
            auto a0 = Clock::now();
            const float clampMin = 0.1f;
            for (int i = 0; i < tasks; ++i) {
                float Pi = sd.newperiods[i];
                float sumOther = 0.0f;
                for (int j = 0; j < tasks; ++j)
                    if (j != i) sumOther += sd.newperiods[j];
                float denom = 1.0f + Pi * sumOther * fabs(u[i]);
                if (denom <= 0.0f) denom = 1.0f;
                sd.newperiods[i] = max(clampMin, Pi / denom);
            }
            auto a1 = Clock::now();
            actOH.push_back(chrono::duration_cast<chrono::nanoseconds>(a1 - a0).count());
        }

        done.store(true);
        producer.join();

        auto m = stats(measOH), c = stats(ctrlOH);
        auto a_raw = stats(actOH);
        pair<double, double> a = { a_raw.first / 1000.0, a_raw.second / 1000.0 };
        cout << setw(8) << tasks
             << setw(15) << fixed << setprecision(2) << m.first
             << setw(15) << m.second
             << setw(15) << c.first
             << setw(15) << c.second
             << setw(15) << a.first
             << setw(15) << a.second << '\n';

        ofstream out("mean_times_"s + to_string(tasks) + ".txt");
        out << "Tasks      = " << tasks << '\n'
            << "Meas us    : mean=" << m.first << ", std=" << m.second << '\n'
            << "Ctrl us    : mean=" << c.first << ", std=" << c.second << '\n'
            << "Act us     : mean=" << a.first << ", std=" << a.second << '\n';
    }

    return 0;
}